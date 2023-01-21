/*
 * Copyright (C) 2004 2009, Magnus Hjorth
 *
 * This file is part of gWaveEdit.
 *
 * gWaveEdit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by        
 * the Free Software Foundation; either version 2 of the License, or  
 * (at your option) any later version.
 *
 * gWaveEdit is distributed in the hope that it will be useful,   
 * but WITHOUT ANY WARRANTY; without even the implied warranty of  
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gWaveEdit; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */

#include <config.h>

#include <gtk/gtk.h>
#include "main.h"
#include "listobject.h"

G_DEFINE_TYPE(ListObject,list_object,GTK_TYPE_OBJECT)

enum { ITEM_ADDED_SIGNAL, ITEM_REMOVED_SIGNAL, ITEM_NOTIFY_SIGNAL, 
       LAST_SIGNAL };
static guint list_object_signals[LAST_SIGNAL] = { 0 };

static void list_object_destroy(GtkObject *obj)
{
     ListObject *lo = LIST_OBJECT(obj);
     list_object_clear(lo,FALSE);
}

static void list_object_class_init(ListObjectClass *klass)
{
     GtkObjectClass *oc = GTK_OBJECT_CLASS(klass);

     oc->destroy = list_object_destroy;
     klass->item_removed = NULL;
     klass->item_added = NULL;

     list_object_signals[ITEM_ADDED_SIGNAL] = 
	  g_signal_new("item-added", G_TYPE_FROM_CLASS(klass), G_SIGNAL_RUN_FIRST,
		       G_STRUCT_OFFSET(ListObjectClass,item_added),
		       NULL, NULL,
		       g_cclosure_marshal_VOID__POINTER, G_TYPE_NONE,
		       1, G_TYPE_POINTER);
     list_object_signals[ITEM_REMOVED_SIGNAL] = 
	  g_signal_new("item-removed", G_TYPE_FROM_CLASS(klass), G_SIGNAL_RUN_FIRST,
		       G_STRUCT_OFFSET(ListObjectClass,item_removed),
		       NULL, NULL,
		       g_cclosure_marshal_VOID__POINTER, G_TYPE_NONE,
		       1, G_TYPE_POINTER);
     list_object_signals[ITEM_NOTIFY_SIGNAL] = 
	  g_signal_new("item-notify", G_TYPE_FROM_CLASS(klass), G_SIGNAL_RUN_FIRST,
		       G_STRUCT_OFFSET(ListObjectClass,item_notify),
		       NULL, NULL,
		       g_cclosure_marshal_VOID__POINTER, G_TYPE_NONE,
		       1, G_TYPE_POINTER);
}

static void list_object_init(ListObject *lo)
{
     lo->list = NULL;
     lo->do_ref = FALSE;
}

ListObject *list_object_new(gboolean do_ref)
{
     ListObject *lo = g_object_new(LIST_OBJECT_TYPE, NULL);
     lo->do_ref = do_ref;
     return lo;
}

ListObject *list_object_new_from_list(GList *list, gboolean do_ref)
{
     ListObject *lo = g_object_new(LIST_OBJECT_TYPE, NULL);
     lo->do_ref = do_ref;
     lo->list = list;
     if (do_ref) {
	  g_list_foreach(list,(GFunc)gtk_object_ref,NULL);
	  g_list_foreach(list,(GFunc)gtk_object_sink,NULL);
     }
     return lo;
}

void list_object_add(ListObject *lo, gpointer ptr)
{
     lo->list = g_list_append(lo->list, ptr);
     if (lo->do_ref) {
	  gtk_object_ref(GTK_OBJECT(ptr));
	  gtk_object_sink(GTK_OBJECT(ptr));
     }
     g_signal_emit(G_OBJECT(lo),list_object_signals[ITEM_ADDED_SIGNAL],0,
		     ptr);
}

gboolean list_object_remove(ListObject *lo, gpointer ptr)
{
     GList *l;
     l = g_list_find(lo->list,ptr);
     if (l == NULL) return FALSE;
     lo->list = g_list_remove_link( lo->list, l );
     g_signal_emit(G_OBJECT(lo),list_object_signals[ITEM_REMOVED_SIGNAL],0,
		     ptr);
     if (lo->do_ref) gtk_object_unref(GTK_OBJECT(ptr));
     g_list_free_1(l);
     return TRUE;
}

void list_object_notify(ListObject *lo, gpointer ptr)
{
     g_signal_emit(G_OBJECT(lo),list_object_signals[ITEM_NOTIFY_SIGNAL],0,
		     ptr);
}

void list_object_clear(ListObject *lo, gboolean do_signal)
{
     GList *l;
     if (do_signal)
	  while (lo->list != NULL) list_object_remove(lo,lo->list->data);
     else {
	  l = lo->list;
	  lo->list = NULL;
	  if (lo->do_ref) g_list_foreach(l,(GFunc)gtk_object_unref,NULL);
	  g_list_free(l);
     }
}

guint list_object_get_size(ListObject *lo)
{
     return g_list_length(lo->list);
}

gpointer list_object_get(ListObject *lo, guint index)
{
     return g_list_nth_data(lo->list, index);
}

void list_object_foreach(ListObject *lo, GFunc func, gpointer user_data)
{
     g_list_foreach(lo->list,func,user_data);
}
