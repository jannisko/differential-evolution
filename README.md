## Usage

```python
import tensorflow as tf

from differential_evolution import DifferentialEvolution
from differential_evolution.losses import create_huber
from differential_evolution.models import create_quadratic

x = [1,2,3,4,5,6,7,8]
y = [1,4,9,16,25,36,49,64]

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