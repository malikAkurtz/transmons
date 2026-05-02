class Branch():
    global_id = 0

    def __init__(self):
        pass


class Capacitor(Branch):
    def __init__(self, capacitance: float):
        self._id = Branch.global_id
        Branch.global_id += 1

        self.capacitance = capacitance

class Inductor(Branch):
    def __init__(self, inductance: float):
        self._id = Branch.global_id
        Branch.global_id += 1

        self.inductance = inductance

class JosephsonElement(Branch):
    def __init__(self, josephson_energy: float):
        self._id = Branch.global_id
        Branch.global_id += 1

        self.josephson_energy = josephson_energy
