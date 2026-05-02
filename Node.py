class Node():
    global_id = 0

    def __init__(self):
        self._id = Node.global_id
        Node.global_id += 1
