from django.urls import path
from . import views

urlpatterns = [
     path("",views.index,name="index"),
     path("doc/<str:doc>",views.doc_by_id,name='doc_by_id'),
     path("search_doc",views.search_doc,name='search_doc')
       ]