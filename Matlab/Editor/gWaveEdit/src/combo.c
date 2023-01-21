/*
 * Copyright (C) 2004 2006 2009, Magnus Hjorth
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


#include "combo.h"

G_DEFINE_TYPE(Combo,combo,GTK_TYPE_COMBO_BOX_TEXT)

enum { CHANGED_SIGNAL, LAST_SIGNAL };
static guint combo_signals[LAST_SIGNAL] = { 0 };

static void combo_size_request(GtkWidget *widget, GtkRequisition *req)
{
     Combo *obj = COMBO(widget);
     GTK_WIDGET_CLASS(combo_parent_class)->size_request(widget,req);
     if (obj->max_request_width >= 0 && req->width > obj->max_request_width)
	  req->width = obj->max_request_width;
}


static void combo_changed(GtkComboBox *combo)
{
    g_signal_emit (G_OBJECT(combo),combo_signals[CHANGED_SIGNAL],0);

    if (GTK_COMBO_BOX_CLASS(combo_parent_class)->changed) {
        GTK_COMBO_BOX_CLASS(combo_parent_class)->changed(combo);
    }
}


static void combo_class_init(ComboClass *klass)
{
     GtkWidgetClass *wc = GTK_WIDGET_CLASS(klass);
     GTK_COMBO_BOX_CLASS(klass)->changed = combo_changed;
     klass->selection_changed = NULL;
     wc->size_request = combo_size_request;
     combo_signals[CHANGED_SIGNAL] = 
        g_signal_new ("selection_changed",
                        G_TYPE_FROM_CLASS(klass),
                        G_SIGNAL_RUN_LAST,
                        G_STRUCT_OFFSET (ComboClass,selection_changed),
                        NULL, NULL,
                        g_cclosure_marshal_VOID__VOID, G_TYPE_NONE, 0);
}


static void combo_init(Combo *obj)
{
    obj->max_request_width = -1;
#if !GTK_CHECK_VERSION(2, 24, 0)
    // - for GTK < 2.24 the combo is still a simple ComboBox
    // - combo_new(): gtk_combo_box_new_text() doesn't work (needs g_object_new)
    //   a lot of GObject errors and the selection_changed signal doesn't work
    GtkWidget * combo_box = GTK_WIDGET (obj);
    GtkCellRenderer * cell;
    GtkListStore * store;

    store = gtk_list_store_new (1, G_TYPE_STRING);
    gtk_combo_box_set_model (GTK_COMBO_BOX (combo_box), GTK_TREE_MODEL (store));
    g_object_unref (store);

    cell = gtk_cell_renderer_text_new ();
    gtk_cell_layout_pack_start (GTK_CELL_LAYOUT (combo_box), cell, TRUE);
    gtk_cell_layout_set_attributes (GTK_CELL_LAYOUT (combo_box), cell,
                                    "text", 0, NULL);
#endif
}

void combo_remove_item(Combo *combo, int item_index)
{
     int i;
     i = gtk_combo_box_get_active (GTK_COMBO_BOX (combo));
     g_assert(i != item_index);
     gtk_combo_box_text_remove(GTK_COMBO_BOX_TEXT(combo),item_index);
}

GtkWidget *combo_new(void)
{
    return (GtkWidget *) g_object_new(COMBO_TYPE, NULL);
}

void combo_set_max_request_width(Combo *c, int width)
{
     c->max_request_width = width;
}

// =============================================================

void w_gtk_glist_to_combo (GtkComboBox *combo, GList *strings, int default_index)
{
    GList * list;
    char * text;
    int len = 0;
    gtk_combo_box_text_remove_all (GTK_COMBO_BOX_TEXT (combo));
    if (!strings) {
        return; // nothing to add
    }
    for (list = strings;  list;  list = list->next)
    {
        text = (char *) list->data;
        gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (combo), text);
        len++;
    }
    if (default_index >= len || default_index < 0) {
        default_index = 0;
    }
    gtk_combo_box_set_active (GTK_COMBO_BOX (combo), default_index);
}


void w_gtk_strv_to_combo (GtkComboBox *combo, char **strv, int default_index)
{
    int i;
    GtkListStore * store;
    GtkTreeIter iter;
    store = GTK_LIST_STORE (gtk_combo_box_get_model (combo));
    gtk_list_store_clear (store);
    //gtk_combo_box_text_remove_all (GTK_COMBO_BOX_TEXT (combo));
    if (!strv || !*strv) {
        return; /* nothing to add */
    }
    for (i = 0; strv[i]; i++) {
        gtk_list_store_append (store, &iter);
        gtk_list_store_set (store, &iter, 0, strv[i], -1);
        //gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (combo), strv[i]);
    }
    if (default_index >= i || default_index < 0) {
        default_index = 0;
    }
    gtk_combo_box_set_active (GTK_COMBO_BOX (combo), default_index);
}
