from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),  # Trang chính
    path("video_feed/", views.video_feed, name="video_feed"),  # Video stream
]
