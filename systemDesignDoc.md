# Modulos

## Reloj (Simulation)

Todo se simula en tiempo real mediante un sistema de **ticks** (pasos de tiempo discretos). El sistema es completamente dinámico: si evolucionas un tick, el reloj global (`current_time`) avanza una cantidad fija de segundos (`step_size`). En cada paso, pasa un segundo/minuto en la calle para los conductores, en el restaurante para los pedidos y para los usuarios. Este objeto actúa como el orquestador central (Simulation) que sincroniza a todos los agentes.

## Restaurantes

Es responsable de completar pedidos de los usuarios. Se trata de un punto fijo en el mapa con: capacidad finita, pedidos que se tardan un tiempo determinado en ser completados y popularidad dinámica. Una instancia de restaurante interactúa con muchas instancias de conductores.

Atributos:
* **Ubicación:** Coordenadas (x, y) para ruteo real.
* **Rating:** (float, dinámico) Afecta la demanda.
* **Avg Prep Time:** Parámetro para la distribución de tiempos de cocina.
* **Capacidad:** Límite máximo de pedidos simultáneos que la cocina puede procesar (`capacity`). Si la lista de `active_orders` iguala a la capacidad, el restaurante rechaza pedidos.
* **Enabled:** Estado lógico que determina si el restaurante está abierto o aceptando órdenes.
**Radio de servicio** Necesario para derterminar si un usuario es capaz de seleccionar un restaurante o no basado en sus ubicaciones relativas. 

Dinámica de preparación:
Los pedidos tienen un tiempo de preparación finito generado estocásticamente (distribución exponencial). Un pedido $p_i$ pasa de estado `PREPARING` a `READY` cuando la diferencia entre el tiempo actual y el de aceptación es mayor al tiempo de preparación calculado. La capacidad solo se libera cuando un conductor recoge el pedido (`PICKED_UP`), no cuando se termina de cocinar.

## Ordenes

La unidad más atómica del sistema. La genera un usuario, se envía a un restaurante, se asigna a un conductor y se cierra solo cuando fue entregada.

Atributos:
* **ID:** Identificador único generado por el contador global de la simulación.
* **Referencias:** Mantiene vínculos al Restaurante, Usuario y Conductor.
* **Timestamps:** Registra el `start_time`, `prep_time` y `ready_time`.
* **Estados:**
    * `PREPARING`: En cocina.
    * `READY`: Listo en mostrador, esperando conductor.
    * `PICKED_UP`: En tránsito con el repartidor.
    * `DELIVERED`: Entregado satisfactoriamente.

## Conductores

Agentes móviles que buscan maximizar sus ganancias. Son entidades autónomas que deciden si aceptan o no un pedido basándose en su propia lógica de negocio.

Atributos:
* **Ubicación Dinámica:** Cambia tick a tick según la velocidad y el grafo de ruteo real.
* **Función de Utilidad:** Cálculo de ganancia neta (Pago - Costos de traslado - Tiempo de espera).
* **Máquina de Estados:**
    * `IDLE`: Disponible para recibir ofertas.
    * `PICKING_UP`: Viajando hacia el restaurante para recolectar un pedido `READY`.
    * `DELIVERING`: Viajando hacia la ubicación del usuario con el pedido en mano.

## Usuarios 
Crean ordenes. Para que una orden pueda ser creada, el 