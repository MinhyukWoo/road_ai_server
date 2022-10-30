from django.urls import path
from ai_rest.views import PredictView

urlpatterns = [path(route="predict", view=PredictView.as_view(), name=PredictView.name)]
