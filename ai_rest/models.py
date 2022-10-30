from django.db import models

# Create your models here.
import os


def image_upload_to(instance, filename):
    return "image.jpg"


class PredictImage(models.Model):
    id = models.BigAutoField(primary_key=True)
    image = models.ImageField(null=False, upload_to=image_upload_to)
