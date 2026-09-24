import math


class CentroidTracker:

    def __init__(
        self,
        max_distance=60,
        max_disappeared=10
    ):
        """
        Seguimiento simple por distancia entre centroides.
        """

        self.next_id = 1

        self.objects = {}

        self.disappeared = {}

        self.max_distance = max_distance
        self.max_disappeared = max_disappeared


    # Registra un objeto nuevo.

    def register(self, centroid):

        object_id = self.next_id

        self.objects[object_id] = centroid

        self.disappeared[object_id] = 0

        self.next_id += 1

        return object_id


    # Elimina un objeto.

    def deregister(self, object_id):

        if object_id in self.objects:
            del self.objects[object_id]

        if object_id in self.disappeared:
            del self.disappeared[object_id]


    # Calcula la distancia entre dos puntos.

    def distance(self, point1, point2):

        x1, y1 = point1
        x2, y2 = point2

        return math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )


    # Actualiza el seguimiento.

    def update(self, detections):

        # Convierte detecciones en centroides.

        input_centroids = []

        for detection in detections:

            centroid = (
                detection["cx"],
                detection["cy"]
            )

            input_centroids.append(
                centroid
            )


        # Si no hay detecciones, marca objetos como perdidos.

        if len(input_centroids) == 0:

            for object_id in list(
                self.disappeared.keys()
            ):

                self.disappeared[object_id] += 1

                if (
                    self.disappeared[object_id]
                    > self.max_disappeared
                ):

                    self.deregister(
                        object_id
                    )

            # Conserva las pistas internamente para recuperarlas si vuelven,
            # pero no las reporta como visibles al contador.
            return {}


        # Si aún no hay objetos, los registra.

        if len(self.objects) == 0:

            visible_objects = {}

            for centroid in input_centroids:

                object_id = self.register(
                    centroid
                )

                visible_objects[object_id] = centroid

            return visible_objects


        # Compara centroides previos con los nuevos.

        object_ids = list(
            self.objects.keys()
        )

        object_centroids = list(
            self.objects.values()
        )


        used_detections = set()
        used_objects = set()
        visible_objects = {}


        # Ordena globalmente las parejas por distancia. Así la asociación no
        # depende del orden de los IDs cuando dos personas están cerca.

        candidate_pairs = []

        for object_index, old_centroid in enumerate(object_centroids):

            for detection_index, new_centroid in enumerate(input_centroids):

                d = self.distance(old_centroid, new_centroid)

                if d <= self.max_distance:
                    candidate_pairs.append(
                        (d, object_index, detection_index)
                    )

        candidate_pairs.sort(key=lambda pair: pair[0])

        for _, object_index, detection_index in candidate_pairs:

            object_id = object_ids[object_index]

            if (
                object_id in used_objects
                or detection_index in used_detections
            ):
                continue

            centroid = input_centroids[detection_index]
            self.objects[object_id] = centroid
            self.disappeared[object_id] = 0
            visible_objects[object_id] = centroid

            used_detections.add(detection_index)
            used_objects.add(object_id)


        # Marca objetos que desaparecieron.

        for object_id in object_ids:

            if object_id not in used_objects:

                self.disappeared[
                    object_id
                ] += 1


                if (
                    self.disappeared[object_id]
                    > self.max_disappeared
                ):

                    self.deregister(
                        object_id
                    )


        # Registra nuevas detecciones.

        for detection_index, centroid in enumerate(
            input_centroids
        ):

            if detection_index not in used_detections:

                object_id = self.register(
                    centroid
                )

                visible_objects[object_id] = centroid


        return visible_objects
