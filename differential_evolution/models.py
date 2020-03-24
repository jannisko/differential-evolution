
def create_quadratic(input_vals_x, a, b, c):
    def quadratic():
        return a * input_vals_x*input_vals_x + b* input_vals_x + c
    return quadratic

def create_linear(input_vals_x, b, c):
    def linear():
        return b * input_vals_x + c
    return linear