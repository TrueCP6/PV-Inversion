from variator import Variator
import numpy as np
import matplotlib.pyplot as plt

def main():
    vary = Variator()

    x = np.linspace(273.15, 273.15+30, 5)
    y = [vary._eval_point({'theta_bar_bottom' : pt}).max_surf_wind_speed() for pt in x]

    if vary.comm.rank == 0:
        plt.plot(x,y)
        plt.show()

if __name__ == "__main__":
    main()