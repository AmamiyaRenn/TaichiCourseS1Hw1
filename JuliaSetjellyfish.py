'''
Description: 一个像水母的juliaSet
Author: Amamiya
Date: 2022-04-11 12:33:05
TechChangeTheWorld
FromWuhanUniversity
'''
from math import pi
import taichi as ti

ti.init(arch=ti.gpu)

n = 640
pixels = ti.Vector.field(3, dtype=float,
                         shape=(n * 2, n))  # 定义像素场为（浮点型，（n*2, n）的矩形）


@ti.func  #只可以被kernel调用且强制内联（不要进行递归）
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallized over all pixels
        c = ti.Vector([(1 + ti.sin(t)) * 0.285, (1 + ti.cos(t)) * 0.1])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        rgb = ti.Vector([0, 1, 1])
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1  # 灰度量
        pixels[i, j] = (1 - iterations * 0.02) * rgb


gui = ti.GUI("Julia Set", res=(n * 2, n))
i = 0
flag = 0
while 1:
    if (flag == 0):
        i -= 1
        if (i * 0.02 <= 0.2):
            flag = 1
    else:
        i += 1
        if (i * 0.02 > (pi * 1.2)):
            flag = 0

    paint(i * 0.02)
    gui.set_image(pixels)
    gui.show()