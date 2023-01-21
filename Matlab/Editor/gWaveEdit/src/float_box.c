/*
 * Copyright (C) 2002 2003 2004 2005 2011, Magnus Hjorth
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

#include <stdlib.h>
#include <gtk/gtk.h>
#include "um.h"
#include "float_box.h"
#include "main.h"

G_DEFINE_TYPE(Floatbox,floatbox,GTK_TYPE_ENTRY)

enum {
     NUMCHANGED_SIGNAL,
     LAST_SIGNAL
};


static guint floatbox_signals[LAST_SIGNAL] = { 0 };

static void floatbox_update_text(Floatbox *box)
{
     char e[30];
     format_float(box->val, e, sizeof(e));
     gtk_entry_set_text(GTK_ENTRY(box),e);
}

static void floatbox_activate(GtkEntry *editable)
{
     float f;
     char *c,*d;
     c=(char *)gtk_entry_get_text(GTK_ENTRY(editable));
     f=strtod(c,&d);
     if (*d==0)
	  floatbox_set(FLOATBOX(editable),f);
     else
	  floatbox_update_text(FLOATBOX(editable));
     if (GTK_ENTRY_CLASS(floatbox_parent_class)->activate)
          GTK_ENTRY_CLASS(floatbox_parent_class)->activate(editable);
}

static gint floatbox_focus_out(GtkWidget *widget, GdkEventFocus *event)
{
     char *c,*d;
     float f;
     Floatbox *b = FLOATBOX(widget);
     c=(char *)gtk_entry_get_text(GTK_ENTRY(widget));
     f=strtod(c,&d);
     if (*d==0 && b->adj!=NULL && f>=gtk_adjustment_get_lower(b->adj) &&
	 f<=gtk_adjustment_get_upper(b->adj)) {
	  gtk_adjustment_set_value(b->adj,f);
     }
     return GTK_WIDGET_CLASS(floatbox_parent_class)->focus_out_event(widget,event);
}

static void floatbox_class_init(FloatboxClass *klass)
{
     GtkWidgetClass *wc = GTK_WIDGET_CLASS(klass);

     GTK_ENTRY_CLASS(klass)->activate = floatbox_activate;
     wc->focus_out_event = floatbox_focus_out;
     klass->numchange=NULL;

     floatbox_signals[NUMCHANGED_SIGNAL] = 
	  g_signal_new("numchanged", G_TYPE_FROM_CLASS(klass),
		       G_SIGNAL_RUN_FIRST,
		       G_STRUCT_OFFSET(FloatboxClass,numchange),
		       NULL, NULL,
		       g_cclosure_marshal_VOID__FLOAT,G_TYPE_NONE,1,
		       G_TYPE_FLOAT);
}

static void floatbox_init(Floatbox *fbox)
{
     gtk_entry_set_width_chars(GTK_ENTRY(fbox),10);
     fbox->adj = NULL;
}

void floatbox_set(Floatbox *box, float val)
{
     if (box->val == val) return;
     if (box->adj != NULL && 
	 val >= gtk_adjustment_get_lower(box->adj) && 
	 val <= gtk_adjustment_get_upper(box->adj)) {

	  gtk_adjustment_set_value(box->adj, val);
	  return;
     }
     box->val=val;
     floatbox_update_text(box);
     g_signal_emit(G_OBJECT(box),floatbox_signals[NUMCHANGED_SIGNAL],0,box->val);
}

GtkWidget *floatbox_new(float val)
{
     Floatbox *box;
     box=g_object_new(FLOATBOX_TYPE, NULL);
     box->val = val-1.0; /* To force update */
     floatbox_set(box,val);
     return GTK_WIDGET(box);
}

gboolean floatbox_check(Floatbox *box)
{
     gfloat f;
     char *c,*d;
     c=(char *)gtk_entry_get_text(GTK_ENTRY(box));
     f=strtod(c,&d);
     if (*d==0) {
	  floatbox_set(box,f);
	  return FALSE;
     } else {
	  d = g_strdup_printf(_("'%s' is not a number!"),c);
	  user_error(d);
	  g_free(d);
	  return TRUE;
     }
}

gboolean floatbox_check_limit(Floatbox *box, float lowest, float highest,
			      gchar *valuename)
{
     gfloat f;
     char *c,*d;
     c=(char *)gtk_entry_get_text(GTK_ENTRY(box));
     f=strtod(c,&d);
     if (*d==0 && f >= lowest && f <= highest) {
	  floatbox_set(box,f);
	  return FALSE;
     } else {
	  d = g_strdup_printf(_("Value for '%s' must be a number between %f and "
			      "%f"),valuename,lowest,highest);
	  user_error(d);
	  g_free(d);
	  return TRUE;
     }     
}


static void floatbox_adj_changed(GtkAdjustment *adjustment, gpointer user_data)
{
     Floatbox *box = FLOATBOX(user_data);
     box->val = box->adj->value;
     floatbox_update_text(box);
     g_signal_emit(G_OBJECT(box),floatbox_signals[NUMCHANGED_SIGNAL],0,
		     box->val);     
}

GtkWidget *floatbox_create_scale(Floatbox *box, float minval, float maxval)
{
     GtkWidget *w;
     GtkRequisition req;
     if (box->adj == NULL) {	  
	  box->adj = GTK_ADJUSTMENT(gtk_adjustment_new(minval,minval,
						       maxval+
						       (maxval-minval)/10.0,
						       (maxval-minval)/100.0,
						       (maxval-minval)/4.0,
						       (maxval-minval)/10.0));
	  g_signal_connect(G_OBJECT(box->adj),"value_changed",
			     G_CALLBACK(floatbox_adj_changed),box);
	  gtk_adjustment_set_value(box->adj,box->val);
     }
     w = gtk_hscale_new(box->adj);
     gtk_widget_size_request(w,&req);
     gtk_widget_set_size_request(w,req.width*5,req.height);
     return w;
}
