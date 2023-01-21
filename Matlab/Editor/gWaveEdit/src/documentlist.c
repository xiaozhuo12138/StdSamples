/*
 * Copyright (C) 2002 2003 2004 2005 2009, Magnus Hjorth
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


#include "documentlist.h"

G_DEFINE_TYPE(DocumentList,document_list,COMBO_TYPE)

enum { CHANGED_SIGNAL, LAST_SIGNAL };
static guint document_list_signals[LAST_SIGNAL] = { 0 };
static gboolean updating = FALSE;

static void document_list_changed(Combo *combo)
{
     Document *d=NULL;
     DocumentList *dl = DOCUMENT_LIST(combo);
     int selindex = gtk_combo_box_get_active (GTK_COMBO_BOX (combo));

     if (updating) return;
     d = DOCUMENT (list_object_get(document_objects, selindex));
     if (d != dl->selected) {
	  dl->selected = d;
	  memcpy(&(dl->format),&(d->chunk->format),sizeof(Dataformat));
	  g_signal_emit(G_OBJECT(dl),
			  document_list_signals[CHANGED_SIGNAL],0);
     }

     if (COMBO_CLASS(document_list_parent_class)->selection_changed)
	  COMBO_CLASS(document_list_parent_class)->selection_changed(combo);
}

static void document_list_addnotify(ListObject *lo, gpointer item, 
				      gpointer user_data)
{
     Document *d = DOCUMENT(item);
     DocumentList *dl = DOCUMENT_LIST(user_data);
     if (dl->selected == NULL) {
	  dl->selected=d;
	  gtk_widget_set_sensitive(GTK_WIDGET(dl),TRUE);
     }
     document_list_setup(dl,dl->selected);     
     if (d == dl->selected) {	  
	  g_signal_emit(G_OBJECT(dl),
			  document_list_signals[CHANGED_SIGNAL],0);
     }
}

static void document_list_remove(ListObject *lo, gpointer item,
				   gpointer user_data)
{
     Document *w = DOCUMENT(item);
     DocumentList *mwl = DOCUMENT_LIST(user_data);
     if (w == mwl->selected) {
	  if (list_object_get_size(document_objects) == 0) {
	       /* We set the selected item to NULL and dim the widget, but 
		* no signal is sent out. */
	       mwl->selected = NULL;
	       gtk_widget_set_sensitive(GTK_WIDGET(mwl),FALSE);
	       return;
	  }
	  mwl->selected = DOCUMENT(list_object_get(document_objects,0));
	  document_list_setup(mwl,mwl->selected);
	  g_signal_emit(G_OBJECT(mwl),
			  document_list_signals[CHANGED_SIGNAL],0);
     } else
	  document_list_setup(mwl,mwl->selected);

}

static void document_list_init(DocumentList *mwl)
{
     /* Most initialization is done in document_list_setup */
     combo_set_max_request_width(COMBO(mwl),350);
     mwl->selected = NULL;
     g_signal_connect_object(document_objects, "item_added",
			     G_CALLBACK(document_list_addnotify), mwl, 0);
     g_signal_connect_object(document_objects, "item_removed",
			     G_CALLBACK(document_list_remove), mwl, 0);
     g_signal_connect_object(document_objects, "item_notify",
			     G_CALLBACK(document_list_addnotify), mwl, 0);
}

static void document_list_class_init(DocumentListClass *klass)
{
     klass->document_changed = NULL;
     COMBO_CLASS(klass)->selection_changed = document_list_changed;

     document_list_signals[CHANGED_SIGNAL] = 
	  g_signal_new("document_changed", G_TYPE_FROM_CLASS(klass),
	               G_SIGNAL_RUN_FIRST,
		       G_STRUCT_OFFSET(DocumentListClass,document_changed),
		       NULL, NULL,
		       g_cclosure_marshal_VOID__VOID, G_TYPE_NONE, 0);
}

GtkWidget *document_list_new(Document *chosen)
{
     GtkWidget *widget;
     widget = GTK_WIDGET(g_object_new(DOCUMENT_LIST_TYPE, NULL));
     document_list_setup(DOCUMENT_LIST(widget),chosen);
     return widget;     
}

struct setup_func_data {
     GList *lp;
     Document *first;
};

static void document_list_setup_func(gpointer item, gpointer user_data)
{
     struct setup_func_data *sfdp = (struct setup_func_data *)user_data;
     Document *w = DOCUMENT(item);
     if (w->titlename != NULL) {
	  sfdp->lp = g_list_append(sfdp->lp, w->titlename);
	  if (sfdp->first == NULL) sfdp->first = w;
     }
}

void document_list_setup(DocumentList *mwl, Document *chosen)
{   
     gint i;
     struct setup_func_data sfd;
     updating = TRUE;     

     sfd.lp = NULL;
     sfd.first = NULL;
     list_object_foreach(document_objects,document_list_setup_func,&sfd);

     w_gtk_glist_to_combo (GTK_COMBO_BOX (mwl), sfd.lp, 0);

     if (chosen == NULL) 
	  chosen = sfd.first;

     if (chosen) {
	  i = g_list_index(sfd.lp,chosen->titlename);
	  g_assert(i >= 0);
      gtk_combo_box_set_active (GTK_COMBO_BOX (mwl), i);
	  memcpy(&(mwl->format),&(chosen->chunk->format),
		 sizeof(Dataformat));
	  mwl->selected = chosen;
     }
     g_list_free(sfd.lp);     
     updating = FALSE;
}
