from django.db import models

# Create your models here.


class PredictImage(models.Model):
    image = models.ImageField(
        null=False,
    )
