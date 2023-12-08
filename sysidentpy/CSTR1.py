#source:
#https://www.youtube.com/watch?v=6z6nXlmDXSs

#simple model with differential equations:
# da/dt = k2 * c - k1 * a * b                   = 0.002 * c - 0.008 * a * b
# db/dt = k2 * c - k1 * a * b                   = 0.002 * c - 0.008 * a * b
# dc/dt = 2 * k1 * a * b - 2 * k2 * c           = 0.016 * a * b - 0.004 * c


#Gleichungen umgewandelt in diskretes System:
#a[n+1] = a[n] + (k2 * c[n] - k1 * a[n] * b[n]) * dt
#b[n+1] = b[n] + (k2 * c[n] - k1 * a[n] * b[n]) * dt
#c[n+1] = c[n] + (2 * k1 * a[n] * b[n] - 2 * k2 * c[n]) * dt

import numpy

#settings

def simCSTR1(seconds, tstep, n_variables, x0s):
        steps = int(seconds/tstep)
        data = numpy.zeros((steps, n_variables),dtype=float)
        ydata = numpy.zeros((steps, n_variables),dtype=float)
        t_0 = 0
        k1 = 0.008
        k2 = 0.002
        a = x0s[0]
        b = x0s[1]
        c = x0s[2]
        t = t_0
        adt = 0
        bdt = 0
        cdt = 0

        # for i in range(steps):
        #         data[i] = (a, b, c)
        #         a_i = a + (k2 * c - k1 * a * b) * tstep
        #         b_i = b + (k2 * c - k1 * a * b) * tstep
        #         c_i = c + (2 * k1 * a * b - k2 * 2 * c) * tstep
        #         a = a_i
        #         b = b_i
        #         c = c_i
        # print(data)


        ## veränderter cstr
        for i in range(steps):
                adt = 0.002 * c - 0.008 * a * b
                bdt = 0.002 * c - 0.008 * a * b
                cdt = 0.016 * a * b - 0.004 * c
                data[i] = (a, b, c)
                ydata[i] = (adt, bdt, cdt)
                a_i = a + (k2 * c - k1 * a * b) * tstep
                b_i = b + (k2 * c - k1 * a * b) * tstep
                c_i = c + (2 * k1 * a * b - k2 * 2 * c * c) * tstep
                a = a_i
                b = b_i
                c = c_i


        print(data)
        print(ydata)

        return data, ydata
