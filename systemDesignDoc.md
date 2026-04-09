# Sistema de Simulacion de Entregas -- Documento de Diseno

Ultima actualizacion: Abril 2026

---

## Arquitectura general

El sistema esta organizado en cuatro capas desacopladas:

```md
Simulation              -- orquestador del tiempo y eventos
Environment             -- red vial y logica de ruteo
agents.py               -- Driver, Order, Restaurant, User
policies/               -- DispatchPolicy, RepositioningPolicy y sus implementaciones
```

La `Simulation` no contiene logica de despacho ni de ruteo. Ambas responsabilidades se delegan a un `Environment` y a politicas intercambiables.

---

## Environment

Centraliza toda la logica espacial. Recibe un `nx.MultiDiGraph` de OpenStreetMap y expone metodos de consulta; no conoce agentes ni logica de negocio.

```python
class Environment:
    graph: nx.MultiDiGraph
    _route_cache:     dict[tuple[int,int], tuple]       # (origin, dest) -> (distance, path)
    _reachable_cache: dict[tuple[int,float], dict]      # (origin, cutoff_m) -> {node: dist}
```

### Metodos

`get_route(origin, destination)` -- ruta mas corta con peso `length`. Cacheada permanentemente. Retorna `(None, None)` si no existe camino.

`get_reachable(origin, cutoff_m)` -- nodos alcanzables dentro de `cutoff_m` metros. Sin cache.

`get_reachable_cached(origin, cutoff_m)` -- version cacheada. Valida unicamente para grafos estaticos. Usada por el despachador y por `generate_orders` para acelerar consultas repetidas.

`get_edge_data(u, v)` -- datos del primer arco paralelo entre `u` y `v`.

`get_node_coords(node)` -- retorna `(lon, lat)`.

---

## Simulation

Orquestador delgado. Posee el reloj, los registros de agentes y la agenda de eventos futuros.

### Constructor

```python
Simulation(
    env:                  Environment,
    dispatch_policy:      DispatchPolicy       = HungarianPolicy(),
    repositioning_policy: RepositioningPolicy  = StaticPolicy(),
    step_size:            float = 10,       # segundos por tick
    dispatch_interval:    float = 15,       # segundos entre despachos batch
    start_hour:           float = 0.0       # hora de inicio (0-23)
)
```

### Registros de agentes

```md
restaurants: dict[int, Restaurant]
users:       dict[int, User]
drivers:     dict[int, Driver]
orders:      dict[int, Order]
```

### Estado del despachador

```md
pending_orders:   deque[int]    -- FIFO de order IDs en espera de conductor
_pending_set:     set[int]      -- mismo conjunto, para lookup O(1)
idle_drivers:     set[int]      -- conductores disponibles
_active_user_ids: set[int]      -- usuarios con orden activa (impide duplicados)
```

### Reloj simulado

```python
current_time: float           # segundos desde el inicio de la simulacion

@property wall_clock_hour     # hora del dia simulada (0-24 float)
@property wall_clock_display  # string legible, ej. "Day 2  14:35:10"
```

La hora del dia se calcula como `(start_hour + current_time / 3600) mod 24`.

### Sistema de eventos programados

La simulacion mantiene una agenda de eventos futuros:

```python
_schedule: list[tuple[float, str, Any]]   # (trigger_time, event_type, payload)
```

Los eventos se procesan al inicio de cada tick mediante `_process_schedule()`.

Tipos de evento soportados:

| event_type       | payload   | Efecto                                                                           |
|------------------|-----------|----------------------------------------------------------------------------------|
| disable_driver   | driver_id | Marca available = False. Si esta IDLE sin cola, se elimina de idle_drivers.      |
| enable_driver    | driver_id | Marca available = True. Si esta IDLE, se agrega a idle_drivers.                  |
| add_driver       | Driver    | Registra un conductor nuevo en la simulacion.                                    |

### Ciclo de un tick

Cada llamada a `run_tick()` ejecuta en orden:

1. `_process_schedule()` -- dispara los eventos cuyo trigger_time <= current_time.
2. Actualizacion de restaurantes -- transiciones PREPARING -> READY cuando current_time >= ready_time.
3. `update_position()` de cada conductor disponible -- avanza sobre el grafo y emite DriverEvents.
4. `_handle_driver_events()` -- aplica timestamps y efectos de restaurante.
5. `_run_dispatch()` -- ejecuta la politica de despacho si han pasado dispatch_interval segundos.
6. `_run_repositioning()` -- ejecuta la politica de reposicionamiento para conductores ociosos.

---

## Policies

### Interfaces

```python
class DispatchPolicy(ABC):
    def assign(
        idle_driver_locations: dict[int, int],   # {driver_id: nodo_actual}
        pending_orders:        dict[int, int],   # {order_id: nodo_restaurante}
        env: Environment,
    ) -> list[tuple[int, int]]: ...              # [(driver_id, order_id), ...]

class RepositioningPolicy(ABC):
    def reposition(
        idle_driver_locations: dict[int, int],
        env: Environment,
        sim_state: dict,   # claves: 'current_time', 'pending_count', 'restaurant_nodes'
    ) -> dict[int, int]: ...                     # {driver_id: nodo_destino}
```

### GreedyPolicy

FIFO puro. Empareja la siguiente orden pendiente con cualquier conductor disponible sin optimizacion de distancia. Adecuada para despacho por tick sin ventana de batching.

### HungarianPolicy

Matching bipartito por el algoritmo hungaro (`scipy.optimize.linear_sum_assignment`). Minimiza la distancia total conductor -> restaurante sobre todos los pares viables dentro de un radio.

```python
HungarianPolicy(pickup_radius: float = 3000)   # metros
```

Construye una matriz de costos usando `get_reachable_cached`. Las celdas inalcanzables se marcan con `inf` y se excluyen antes de resolver.

### StaticPolicy

No-op. Los conductores permanecen en su posicion actual cuando estan ociosos. Sirve como baseline de reposicionamiento.

### RLPolicy

Socket para un modelo RL futuro. Mientras no haya modelo cargado, cae en comportamiento estatico. La interfaz de entrenamiento (`train`, `_build_obs`) esta declarada pero no implementada.

---

## Agents

### Order

Conecta usuario, restaurante y conductor. Registra todos los timestamps necesarios para calcular metricas.

Atributos:

```md
id, user_id, restaurant_id, driver_id
prep_time, start_time, ready_time
assigned_time, driver_arrival_time, pickup_time, delivered_time
status, route_to_user
```

Ciclo de vida:

```md
PREPARING -> READY -> PICKED_UP -> DELIVERED
```

### Propiedades calculadas

```python
end_to_end_time  = delivered_time - start_time
food_idle_time   = max(0, driver_arrival_time - ready_time)
driver_wait_time = max(0, ready_time - driver_arrival_time)
time_to_assign   = assigned_time - start_time
dispatch_delay   = assigned_time - ready_time
food_wait_time   = max(0, pickup_time - max(ready_time, assigned_time))
```

#### Interpretacion de cada metrica

**`end_to_end_time`** -- experiencia total del cliente desde que coloca la orden hasta que la recibe. Incluye prep time, viaje del driver al restaurante, dwell de pickup, viaje al usuario y dwell de entrega. Es la metrica principal del sistema. Valida.

**`food_idle_time`** -- segundos que la comida estuvo lista en el restaurante antes de que llegara el driver. Mide ineficiencia del lado de la oferta. Cercano a cero en escenarios oversupplied (el driver llega antes de que la comida este lista), alto en undersupplied. Valida.

**`driver_wait_time`** -- segundos que el driver espero en el restaurante antes de que la comida estuviera lista. Mide subutilizacion del driver. Alto en oversupplied, cercano a cero en undersupplied. Junto con `food_idle_time` caracteriza completamente el encuentro restaurante-driver. Valida.

**`dispatch_delay`** -- diferencia entre el momento de asignacion y el momento en que la comida estuvo lista (`assigned_time - ready_time`). Negativo significa que el driver fue asignado antes de que la comida terminara, lo cual es el resultado optimo. Positivo indica que la comida espero sin driver asignado, senalizando escasez local. Solo discrimina entre escenarios en el caso undersupplied; en balanced y oversupplied colapsa a un valor trivialmente negativo (aproximadamente `dispatch_interval - prep_time`) porque siempre hay drivers ociosos disponibles. Valida parcialmente.

**`time_to_assign`** -- segundos desde que se coloca la orden hasta que se asigna un driver (`assigned_time - start_time`). Incluye el prep time en la medicion, por lo que no es una medida de responsividad del despachador. En escenarios con exceso de oferta colapsa al valor del dispatch interval (~10s). No es util como metrica operacional; se mantiene por trazabilidad.

**`food_wait_time`** -- definida como `max(0, pickup_time - max(ready_time, assigned_time))`. Incluye el dwell time del driver en el restaurante (~3.5 min siempre presente), lo que impide distinguir entre espera real y tiempo de handoff normal. No es limpia como metrica operacional. Reemplazada funcionalmente por `food_idle_time` y `driver_wait_time`. Se mantiene por compatibilidad con el ledger existente.

**`order_rating`** -- rating sintetico de 1-5 estrellas generado post-entrega con probabilidad `p_rate = 0.20`. Funcion de `end_to_end_time` y `prior_rating`. Util para analisis de fairness entre restaurantes y zonas, no como ground truth de calidad de servicio.

```python
end_to_end_time  = delivered_time - start_time
food_wait_time   = max(0, pickup_time - max(ready_time, assigned_time))
time_to_assign   = assigned_time - start_time
dispatch_delay   = assigned_time - ready_time
```

`dispatch_delay` mide cuanto tiempo despues de que la comida estuvo lista se asigno un conductor.
Es una senal del balance entre oferta y demanda:

- Negativo: el conductor fue asignado antes de que la comida estuviera lista. El conductor
  viajo al restaurante y espero en el mostrador. Este es el resultado optimo -- la comida
  no tiene tiempo de espera ocioso despues de su preparacion.
- Cercano a cero: el conductor llego aproximadamente cuando la comida estuvo lista.
- Positivo: la comida termino y estuvo sin asignar, esperando a que un conductor estuviera
  disponible. Indica escasez local de conductores en ese momento.

A nivel de flota, la distribucion del signo refleja el balance oferta/demanda a lo largo
de la ventana de simulacion. Un escenario con exceso de conductores mostrara una distribucion
mayormente negativa. Un escenario balanceado estara centrado cerca de cero con ambas colas.
Un escenario con escasez de conductores estara sesgado fuertemente hacia valores positivos,
con valores grandes durante las horas pico.

`time_to_assign` (= `assigned_time - start_time`) tambien esta disponible pero incluye
el tiempo de preparacion en la medicion, lo que lo hace inadecuado como metrica de
capacidad de respuesta del despachador. Es util unicamente como duracion bruta desde
la creacion de la orden hasta su asignacion.

### Restaurant

Nodo fijo del grafo. Produce ordenes con tiempos de preparacion distribuidos como lognormal:

```python
sigma    = 0.5
mu       = ln(avg_prep_time) - sigma^2 / 2
prep_time ~ Lognormal(mu, sigma)
```

El metodo `can_accept_order()` verifica capacidad y aplica una probabilidad de rechazo del 1%.

El metodo `_sync_enabled_status()` actua como circuit breaker: deshabilita el restaurante cuando `len(active_orders) >= capacity` y lo re-habilita cuando la carga baja.

La capacidad se libera cuando el conductor recoge el pedido (`remove_order`), no cuando la orden se marca READY.

Atributos:

```python
location, capacity, avg_prep_time, service_radius, rating
active_orders: list[Order]
enabled: bool
```

### User

Agente estatico. Almacena `user_id` y `location` (nodo del grafo).

### Driver

Agente movil que navega sobre aristas del grafo mediante interpolacion continua. Completamente desacoplado de `Simulation`: no llama de vuelta al simulador, sino que emite `DriverEvent`s que la simulacion observa y reacciona.

#### Estados operativos

```md
IDLE
PICKING_UP  -> PICKUP_SERVICE   (al llegar al restaurante)
DELIVERING  -> DROPOFF_SERVICE  (al llegar al usuario)
```

Los estados PICKUP_SERVICE y DROPOFF_SERVICE modelan el tiempo de dwell en cada punto antes de que la orden cambie de estado.

Tiempos de dwell:

- Pickup: Lognormal(ln(210), 0.45), media aprox. 3.5 min. Fuente: Grubhub MDRP (Reyes et al. 2018).
- Dropoff: mezcla de dos lognormales:
  - 30% handoff rapido -- Lognormal(ln(120), 0.35), media aprox. 2 min.
  - 70% acceso a edificio -- Lognormal(ln(360), 0.45), media aprox. 6 min.
  - Fuente: Zheng et al. 2022, ajustado para Zona Tec Monterrey.

#### Velocidad dinamica

La velocidad se actualiza en cada asignacion de orden segun la hora del dia simulada. Basada en datos TomTom Traffic Index 2025, Monterrey:

| Franja horaria         | Velocidad media |
|------------------------|-----------------|
| Pico mediodia (12-14h) | 31 km/h         |
| Congestion (18-19h)    | 22 km/h         |
| Cena aflojando (19-21h)| 30 km/h         |
| Nocturno (22h-6h)      | 50 km/h         |
| Diurno general         | 32 km/h         |

Se muestrea con ruido gaussiano de desviacion estandar del 10%. Minimo forzado a 4.5 m/s.

#### Disponibilidad

```python
available: bool
```

Cuando `available = False` y el conductor termina su cola, no vuelve a `idle_drivers`. Este mecanismo es el que usan los eventos de turno para retirar conductores gradualmente sin interrumpir entregas en curso.

#### Cola de ordenes

```python
order_queue: deque[Order]
```

Los conductores aceptan multiples ordenes en cola FIFO. Al terminar una entrega procesan automaticamente la siguiente en cola.

#### DriverEvents emitidos

```md
ARRIVED_AT_RESTAURANT
PICKUP_COMPLETE
ARRIVED_AT_USER
DROPOFF_COMPLETE
WENT_IDLE
```

---

## Generacion de ordenes

### generate_orders

Las llegadas siguen un proceso de Poisson por tick:

```md
arrivals ~ Poisson(rate_per_minute / 60 * step_size)
```

La eleccion de restaurante sigue un logit multinomial:

```md
utility(r) = 0.6 * rating(r) - 0.3 * distance_km(r)
```

Coeficientes sintetizados de: Brasil delivery fee model (Frontiers 2022) y Ma et al. 2024 (Singapur mixed logit). El cociente implica indiferencia entre +1 estrella y +2 km, razonable para Monterrey urbano.

Si el usuario seleccionado ya tiene una orden activa (`_active_user_ids`), la llegada se omite.

### get_order_rate

Retorna ordenes/minuto segun la hora del dia. Calibrado para Zona Tec Monterrey:

| Franja  | Ordenes/min |
|---------|-------------|
| 0-6h    | 0.5         |
| 6-10h   | 1.5         |
| 10-11h  | 3.0         |
| 11-12h  | 4.5         |
| 12-14h  | 5.0         |
| 14-17h  | 2.5         |
| 17-19h  | 3.5         |
| 19-22h  | 5.0         |
| 22-24h  | 1.5         |

---

## Turnos de conductores

Modela la dinamica de plataformas gig donde los conductores se autoseleccionan en ventanas de demanda alta.

Estructura de turnos:

```md
Morning cohort (~30% de la flota): 09:00 - 15:00
Evening cohort (~50% de la flota): 17:00 - 23:00
Always-on      (~20% de la flota): activos todo el dia
```

La funcion `schedule_driver_shifts(sim, residential_nodes)` programa eventos `enable_driver` / `disable_driver` en `sim._schedule`. Los tiempos incluyen jitter de +/-15 minutos para evitar cambios masivos simultaneos.

Al inicio, el 80% de los conductores arranca deshabilitado. Se van activando conforme el reloj simulado avanza hacia sus ventanas de turno.

---

## Metricas

`sim.metrics_snapshot()` retorna:

```python
time, dispatch_policy, repositioning_policy
total_orders, orders_by_status
idle_drivers, active_drivers, deactivated_drivers
pending_unassigned
avg_end_to_end_s, avg_food_wait_s, avg_dispatch_delay_s
n_delivered
```

Los resultados completos por orden se exportan a `results.csv` con el ledger de timestamps para analisis offline.

### auditor.py

Modulo de diagnostico que detecta inconsistencias en el ledger:

- Violaciones cronologicas internas (start > assigned, assigned > pickup, pickup > delivered).
- Schedules solapados: un conductor recoge la orden B antes de entregar la orden A.
- Entrega antes de que finalice el tiempo de preparacion.

---

## Warmup

Los notebooks ejecutan `sim.run_until(3600 * WARMUP)` antes del periodo de medicion. Tras el warmup, las ordenes y contadores se resetean manteniendo el estado del sistema (conductores activos, posiciones, cola de eventos). Esto estabiliza la flota y la demanda antes de registrar metricas.

---

## Escenarios de prueba

| Escenario     | Conductores simultaneos objetivo | Pool total |
|---------------|----------------------------------|------------|
| Undersupplied | 50                               | ~72        |
| Balanced      | 150                              | ~215       |
| Oversupplied  | 300                              | ~430       |
