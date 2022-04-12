'''
Description: 改变juliaSet的颜色
Author: Amamiya
Date: 2022-04-12 16:04:04
TechChangeTheWorld
WHUROBOCON_Alright_Reserved
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
        # c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        c = ti.Vector([ti.cos(t), ti.sin(t)]) * 0.7885
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1  # 灰度量
        rgb = ti.Vector([0, 1, 1]) * (z[1] + 1)  # rgb值没搞明白到底是怎么回事
        pixels[i, j] = (1 - iterations * 0.02) * rgb


gui = ti.GUI("Julia Set", res=(n * 2, n))
i = 0
while 1:
    i += 1
    paint(i * 0.015)
    gui.set_image(pixels)
    gui.show()