# Modulos

## Reloj (Simulation)

El sistema funciona como un orquestador central que gestiona la progresión del tiempo mediante **ticks discretos** (`run_tick`). La clase `Simulation` mantiene una visión global de todos los agentes y coordina la evolución del sistema.

### Características principales

#### Infraestructura urbana

El sistema utiliza un grafo vial (`graph`) obtenido de OpenStreetMap.  
Las rutas se calculan utilizando `networkx.shortest_path` con peso `length`, lo que permite trabajar con distancias reales.

#### Registros globales

La simulación mantiene diccionarios indexados por ID para acceso O(1):

```
restaurants: dict[int, Restaurant]
users: dict[int, User]
drivers: dict[int, Driver]
orders: dict[int, Order]
```

#### Route Cache

Para evitar recalcular rutas repetidamente, el sistema mantiene un cache:

```
route_cache[(origin_node, destination_node)] -> (distance, path)
```

Cuando una ruta se calcula por primera vez, su distancia y lista de nodos se almacenan. Las futuras consultas reutilizan esa información.

#### Motor de simulación

Cada `run_tick` ejecuta únicamente la **evolución del estado del sistema**:

1. **Actualización de restaurantes**

Los restaurantes verifican si sus órdenes han terminado de prepararse:

```
PREPARING → READY
```

cuando

```
current_time >= ready_time
```

2. **Actualización de conductores**

Cada conductor ejecuta `update_position`, avanzando sobre el grafo mediante interpolación continua a lo largo de las aristas.

La asignación logística **no ocurre dentro del tick**, sino mediante eventos del sistema.

---

## Restaurantes

Los restaurantes representan nodos fijos del grafo responsables de producir pedidos.

### Atributos

- `location`: nodo del grafo
- `capacity`: número máximo de órdenes simultáneas
- `avg_prep_time`: tiempo promedio de preparación
- `service_radius`: radio máximo de entrega
- `active_orders`: órdenes actualmente en cocina
- `enabled`: estado operativo

### Control de capacidad

Cuando una orden es aceptada:

```
accept_order(order)
```

se agrega a `active_orders`.

La capacidad se libera únicamente cuando el conductor recoge el pedido:

```
remove_order(order)
```

El método `_sync_enabled_status` funciona como un **circuit breaker automático**, deshabilitando el restaurante cuando se alcanza su capacidad máxima.

### Generación de tiempos de preparación

Los tiempos de preparación siguen una distribución exponencial:

```
prep_time ~ Exp(1 / avg_prep_time)
```

Durante cada tick, las órdenes cuyo tiempo de preparación ha finalizado pasan a estado `READY`.

---

## Órdenes

Las órdenes conectan usuarios, restaurantes y conductores.

### Atributos

- `user_id`
- `restaurant_id`
- `driver_id`
- `prep_time`
- `start_time`
- `ready_time`
- `route_to_user`

La ruta `route_to_user` se calcula al momento de crear la orden y contiene la secuencia de nodos desde el restaurante hasta el usuario.

### Ciclo de vida

Las órdenes pasan por los siguientes estados:

```
PREPARING
READY
PICKED_UP
DELIVERED
```

Las órdenes pueden ser asignadas a un conductor incluso si todavía están en estado `PREPARING`.

---

## Conductores

Los conductores son agentes móviles que navegan sobre el grafo.

Cada conductor mantiene una cola FIFO de órdenes:

```
order_queue: deque[Order]
```

### Estados operativos

```
IDLE
PICKING_UP
DELIVERING
```

### Movimiento

El movimiento se realiza sobre las aristas del grafo utilizando interpolación continua basada en la longitud real de cada arista.

Variables internas relevantes:

```
current_route
current_edge
distance_on_edge
coords
```

### Flujo operativo

1. Cuando un conductor recibe una orden, esta se agrega a `order_queue`.

2. Si el conductor está libre (`IDLE`), comienza inmediatamente a desplazarse hacia el restaurante (`PICKING_UP`).

3. Si llega al restaurante antes de que el pedido esté listo, espera en el nodo del restaurante.

4. Cuando el pedido está listo (`READY`), el conductor lo recoge (`PICKED_UP`) y comienza la entrega.

5. Al llegar al usuario, la orden pasa a `DELIVERED` y el conductor procesa la siguiente orden en su cola.

Cuando un conductor termina todas sus órdenes, vuelve al estado `IDLE`.

---

## Lógica de despacho (Event-Driven Dispatch)

El sistema utiliza un **despachador basado en eventos**, evitando escanear todas las órdenes y conductores en cada tick.

La simulación mantiene dos estructuras principales:

```
pending_orders : deque[int]
idle_drivers   : set[int]
```

### Flujo de despacho

**Cuando ocurre uno de estos eventos:**

- Se crea una nueva orden
- Un conductor termina su última entrega

el sistema ejecuta el despachador.

### Algoritmo de asignación

El despachador asigna órdenes mientras existan:

- órdenes pendientes
- conductores disponibles

```
while pending_orders and idle_drivers:
    order ← pending_orders.pop()
    driver ← idle_drivers.pop()
    asignar driver a order
```

### Ventajas

Este enfoque evita escanear todo el sistema en cada tick.

Complejidad aproximada:

```
O(número_de_asignaciones)
```

en lugar de

```
O(número_de_ordenes + número_de_conductores) por tick
```

Esto mejora significativamente la escalabilidad de la simulación cuando el número de agentes crece.