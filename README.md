# Differential Evolution

A (very slow) implementation of the [differential evolution](https://en.wikipedia.org/wiki/Differential_evolution) algorithm in Python using low level Tensorflow functions.


![gif of a quadratic model being optimized](animations/quadratic_models.gif)
![gif of the ackley loss being optimized](animations/ackley_optimization.gif)

Higher quality versions of these animations can be found under [Releases](https://github.com/jannisko/differential-evolution/releases).

## Motivation

I started this project as an assignment for my Tensorflow class. I am sharing this to help anyone with a similar assignment.

At the moment my code is not using Tensorflow to its strenghts, so execution times are very long.


## Getting Started

```bash
git clone git@github.com:jannisko/differential-evolution.git
cd differential-evolution
pip install -r requirements.txt
```

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

### Using your own model
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

### Creating animations

```create_animation.py``` contains a few examples of how to animate your model or loss function. To output as .mp4 you will need [ffmpeg](https://www.ffmpeg.org/). For .gif you will need [imagemagik](https://imagemagick.org/index.php).

- ```animate_models()``` was used for the 2d animation of a quadratic function.
- ```animate_ackley()``` was used for the 3d animation of the ackley function.


## Troubleshooting

- if you run into the error: \
```convert-im6.q16: cache resources exhausted```\
while creating a .gif you might have to [increase imagemakig's memory limit](https://github.com/ImageMagick/ImageMagick/issues/396)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
