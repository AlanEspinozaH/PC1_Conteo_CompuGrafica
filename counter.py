class LineCounter:

    def __init__(
        self,
        line_x,
        margin=10,
        min_seen_frames=5,
        entry_direction="right"
    ):

        # Línea de conteo.
        self.line_x = line_x
        self.margin = margin

        # Requiere varios frames para evitar falsos positivos.
        self.min_seen_frames = min_seen_frames

        # Dirección que cuenta como entrada.
        self.entry_direction = entry_direction

        # Última posición conocida de cada objeto.
        self.last_side = {}

        # Historial de visión por objeto.
        self.seen_frames = {}

        # Objetos ya contados.
        self.counted_ids = set()

        # Totales.
        self.entry_count = 0
        self.exit_count = 0


   
    # DETERMINAR LADO DE LA LÍNEA
    

    def get_side(self, cx):

        # Objeto a la izquierda, derecha o dentro de la zona de tolerancia.
        if cx < self.line_x - self.margin:
            return -1

        if cx > self.line_x + self.margin:
            return 1

        return 0


   
    # ACTUALIZAR CON OBJETOS DEL TRACKER
  

    def update(self, tracked_objects):

        # Eventos de esta iteración.
        events = []

        for object_id, centroid in tracked_objects.items():

            cx, cy = centroid

            # Suma un frame al objeto.
            if object_id not in self.seen_frames:
                self.seen_frames[object_id] = 0

            self.seen_frames[object_id] += 1

            # Lado actual del objeto respecto a la línea.
            current_side = self.get_side(cx)

            # Ignora si está en la zona neutral.
            if current_side == 0:
                continue

            # Guarda la primera posición.
            if object_id not in self.last_side:
                self.last_side[object_id] = current_side
                continue

            previous_side = self.last_side[object_id]

            # Evita repetir conteos.
            if object_id in self.counted_ids:
                self.last_side[object_id] = current_side
                continue

            # Necesita varios frames para validar el cruce.
            if self.seen_frames[object_id] < self.min_seen_frames:
                self.last_side[object_id] = current_side
                continue

            # Detecta movimiento real.
            movement = None

            if previous_side == -1 and current_side == 1:
                movement = "right"

            elif previous_side == 1 and current_side == -1:
                movement = "left"

            # Decide si fue entrada o salida.
            if movement is not None:
                if movement == self.entry_direction:
                    direction = "ENTRADA"
                    self.entry_count += 1
                else:
                    direction = "SALIDA"
                    self.exit_count += 1

                # Marca como contado.
                self.counted_ids.add(object_id)

                events.append({
                    "id": object_id,
                    "direction": direction,
                    "movement": movement,
                    "centroid": (cx, cy)
                })

            # Actualiza la última posición.
            self.last_side[object_id] = current_side

        return events


  
    # TOTAL DE CRUCES
  

    def get_total(self):

        # Total acumulado.
        return (
            self.entry_count +
            self.exit_count
        )