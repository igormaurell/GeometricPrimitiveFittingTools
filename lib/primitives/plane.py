class Plane:
    def __init__(self, parameters: dict):
        self.location = parameters['location']
        self.x_axis = parameters['x_axis']
        self.y_axis = parameters['y_axis']
        self.z_axis = parameters['z_axis']
        self.coefficients = parameters['coefficients']