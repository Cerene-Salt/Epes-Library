import simplegui
import math

# --------------------------------------------------------------------
# 1) CLASSES VETORIAIS E MATRICIAIS
# --------------------------------------------------------------------
class Vec3:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z
    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s)
    def dot(self, o):     return self.x*o.x + self.y*o.y + self.z*o.z
    def cross(self, o):   # produto vetorial
        return Vec3(self.y*o.z - self.z*o.y,
                    self.z*o.x - self.x*o.z,
                    self.x*o.y - self.y*o.x)
    def norm(self):       # norma (comprimento)
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    def normalize(self):
        n = self.norm()
        return Vec3(self.x/n, self.y/n, self.z/n) if n != 0 else Vec3(0, 0, 0)
    def tuple2(self):     # para canvas.draw_image ou draw_polygon
        return [self.x, self.y]

# --------------------------------------------------------------------
# 2) PARÂMETROS GERAIS
# --------------------------------------------------------------------
urls = [
    "https://i.imgur.com/Ioqgatq.jpeg",
    "https://i.imgur.com/YtOZuIx.jpeg",
    "https://i.imgur.com/Nn7zfoV.jpeg"
]
images = [simplegui.load_image(u) for u in urls]

FW, FH = 400, 700

n_images = len(images)
carousel_rad = 120.0
angular_spd = 0.6  # rad/s

focal = 300.0

omega = 2.2
zeta = 0.12
A0 = 0.4

k1 = 1e-6
k2 = 1e-12

crop_amp_x = 0.18
crop_amp_y = 0.12
crop_freq_x = 1.5
crop_freq_y = 1.1

light_dir = Vec3(0.0, 0.0, -1.0).normalize()

t_global = 0.0

# --------------------------------------------------------------------
# 3) FUNÇÕES MATEMÁTICAS
# --------------------------------------------------------------------
def damped_zoom(t):
    return 1.0 + A0 * math.exp(-zeta*t) * math.cos(omega*t)

def perspective_project(v3):
    denom = focal + v3.z
    if denom == 0: denom = 1e-3
    return (focal * v3.x / denom, focal * v3.y / denom)

def radial_distort(x, y):
    r2 = x*x + y*y
    fac = 1 + k1*r2 + k2*(r2*r2)
    return x*fac, y*fac

def oscillating_crop(img, t, phase):
    w, h = img.get_width(), img.get_height()
    dx = crop_amp_x * math.sin(crop_freq_x*t + phase)
    dy = crop_amp_y * math.cos(crop_freq_y*t + phase)
    cw = w * (1 - abs(dx))
    ch = h * (1 - abs(dy))
    cx = w/2 + dx*(w/2 - cw/2)
    cy = h/2 + dy*(h/2 - ch/2)
    return [cx, cy], [cw, ch]

def compute_brightness(normal):
    dp = normal.normalize().dot(light_dir)
    return max(0.2, dp)

# --------------------------------------------------------------------
# 4) DRAW HANDLER
# --------------------------------------------------------------------
def draw(canvas):
    global t_global

    canvas.draw_polygon([[0,0],[FW,0],[FW,FH],[0,FH]], 1, 'Black', 'Black')

    s = damped_zoom(t_global)
    base_size = [300*s, 200*s]

    for i, img in enumerate(images):
        W, H = img.get_width(), img.get_height()
        if W == 0 or H == 0:
            canvas.draw_text("Loading…", [50, 100 + i*220], 24, 'Red')
            continue

        theta = angular_spd * t_global + (2 * math.pi * i / n_images)
        world_pos = Vec3(carousel_rad * math.cos(theta), 0.0, carousel_rad * math.sin(theta))
        normal3 = Vec3(world_pos.x, 0.0, world_pos.z).normalize()

        xp, yp = perspective_project(world_pos)
        yp += 120 + i * 220

        dx, dy = radial_distort(xp, yp - FH/2)
        screen_x = FW/2 + dx
        screen_y = FH/2 + dy

        phase = i * (2 * math.pi / n_images)
        src_center, src_size = oscillating_crop(img, t_global, phase)

        brightness = compute_brightness(normal3)
        draw_size = [base_size[0] * brightness, base_size[1] * brightness]

        canvas.draw_image(img, src_center, src_size, [screen_x, screen_y], draw_size)

    t_global += 0.04

# --------------------------------------------------------------------
# 5) SETUP E INÍCIO
# --------------------------------------------------------------------
frame = simplegui.create_frame("Mega-Galeria 3D & Math-Fu", FW, FH)
frame.set_draw_handler(draw)
frame.start()
