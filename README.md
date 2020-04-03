![gif of a quadratic model being optimized](animations/quadratic_models.gif)
![gif of ackley function being optimized](animations/ackley_optimization.gif)

## Usage

```python
import tensorflow as tf

from differential_evolution import DifferentialEvolution
from differential_evolution.losses import create_huber
from differential_evolution.models import create_quadratic

# define original data
x = [1,2,3,4,5,6,7,8]
y = [1,4,9,16,25,36,49,64]

# define initial variables for the model
a = tf.Variable(0.0)
b = tf.Variable(0.0)
c = tf.Variable(0.0)

model = create_quadratic(x, a, b, c)

loss = create_huber(model, y)

opt = DifferentialEvolution(loss, [a,b,c])

for _ in range(100):
    opt.next_generation()

point, loss = opt.get_best_point()
```

## Creating your own model
```python
x = np.linspace(0,9,10)
y = np.linspace(0,9,10)**2

a = tf.Variable(0.0)
b = tf.Variable(0.0)
c = tf.Variable(0.0)

def logistic_model():
    return c / (1 + a * tf.math.exp(-b * x))

loss = create_huber(logistic_model, y)
```

## Creating animations

```create_animation.py``` contains a few examples of how to animate your model or loss function. To output as .mp4 you will need [ffmpeg](https://www.ffmpeg.org/). For .gif you will need [imagemagik](https://imagemagick.org/index.php).

- ```animate_models()``` was used for the 2d animation of a quadratic function.
- ```animate_loss()``` was used for the 3d animation of the ackley function.


## Troubleshooting

- if you run into the error: \
```convert-im6.q16: cache resources exhausted```\
while creating a .gif you might have to [increase imagemakig's memory limit](https://github.com/ImageMagick/ImageMagick/issues/396)
