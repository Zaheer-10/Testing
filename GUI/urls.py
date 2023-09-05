from . import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

# Configure URL patterns for the web application
urlpatterns = [
    path('', views.index, name='index'),  # Home page
    path('search/', views.search_papers, name='search_papers'), 
    path('recommendations/', views.recommendations, name='recommendations'), 
    path('display/<str:paper_id>/', views.display, name='display'),
    path('about/', views.about, name='about'), 
    path('architecture/', views.architecture, name='architecture'), 

]

# Serve static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
