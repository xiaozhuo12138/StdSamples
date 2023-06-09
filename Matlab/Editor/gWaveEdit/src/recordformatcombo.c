/*
 * Copyright (C) 2009, Magnus Hjorth
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

#include "recordformatcombo.h"

G_DEFINE_TYPE(RecordFormatCombo,record_format_combo,COMBO_TYPE)

enum { FORMAT_CHANGED_SIGNAL, FORMAT_DIALOG_REQUEST_SIGNAL, LAST_SIGNAL };
static guint record_format_combo_signals[LAST_SIGNAL] = { 0 };

static char *choose_format_string = N_("Choose a sample format");

static gboolean nosignal_flag = FALSE;

static void record_format_combo_destroy(GtkObject *obj)
{
     RecordFormatCombo *rfc = RECORD_FORMAT_COMBO(obj);
     
     if (rfc->named_presets) {
	  gtk_object_unref(GTK_OBJECT(rfc->named_presets));
	  rfc->named_presets = NULL;
     }
     if (rfc->nameless_presets) {
	  gtk_object_unref(GTK_OBJECT(rfc->nameless_presets));
	  rfc->nameless_presets = NULL;
     }
     GTK_OBJECT_CLASS(record_format_combo_parent_class)->destroy(obj);
}

static gchar *rf_string(Dataformat *fmt, gchar *name)
{
     gchar *c,*d;
     if (fmt->type == DATAFORMAT_PCM)
	  c = g_strdup_printf(_("%d-bit"),fmt->samplesize*8);
     else if (fmt->samplesize == sizeof(float))
	  c = g_strdup(_("float"));
     else
	  c = g_strdup(_("double"));
     if (name)
	  d = g_strdup_printf(_("%s (%s %s %d Hz)"),name,
			      channel_format_name(fmt->channels),
			      c,fmt->samplerate);
     else 
	  d = g_strdup_printf(_("%s %s %d Hz"),
			      channel_format_name(fmt->channels),
			      c,fmt->samplerate);
     g_free(c);
     return d;
}

static void rebuild_strings(RecordFormatCombo *rfc)
{
     GList *l = NULL, *l2;
     RecordFormat *rf;
     Dataformat *df;
     int idx = -1, count=0;
     if (rfc->current_selection_type == 0) {
	  l = g_list_append(l,g_strdup(_(choose_format_string)));
	  idx = count;
	  count++;
     }
     rfc->named_preset_start = count;
     for (l2 = rfc->named_presets->list; l2 != NULL; l2 = l2->next) {
	  rf = (RecordFormat *)(l2->data);	       
	  l = g_list_append(l,rf_string(&(rf->fmt),rf->name));
	  if (idx < 0 && rfc->current_selection_type == 1 &&
	      !strcmp(rfc->current_selection_name,rf->name) &&
	      dataformat_equal(&(rfc->current_selection_format),
			       &(rf->fmt)))
	       idx = count;
	  count++;
     }
     rfc->nameless_preset_start = count;
     for (l2 = rfc->nameless_presets->list; l2 != NULL; l2 = l2->next) {
	  df = (Dataformat *)(l2->data);
	  l = g_list_append(l,rf_string(df,NULL));
	  if (idx < 0 && rfc->current_selection_type == 2 &&
	      dataformat_equal(&(rfc->current_selection_format),df))
	       idx = count;
	  count ++;
     }
     rfc->custom_start = count;
     if (idx < 0) { 
	  rfc->current_selection_type = 3;
	  l = g_list_append(l,rf_string(&(rfc->current_selection_format),NULL));
	  idx = count;
	  count++;
     }
     rfc->other_start = count;
     if (rfc->show_other)
	  l = g_list_append(l,g_strdup(_("Other format...")));

     w_gtk_glist_to_combo (GTK_COMBO_BOX (rfc), l, idx);
     
     g_list_foreach(l,(GFunc)g_free,NULL);
     g_list_free(l);	  
}

static void record_format_set_main(RecordFormatCombo *rfc, int type, 
				   char *name, Dataformat *fmt,
				   gboolean rebuild)
{
     if (rfc->current_selection_type == type && 
	 (type != 1 || !strcmp(name,rfc->current_selection_name)) &&
	 (type == 0 || dataformat_equal(&rfc->current_selection_format, fmt)))
	  return;
     if (rfc->current_selection_name != NULL) 
	  g_free(rfc->current_selection_name);
     rfc->current_selection_type = type;
     rfc->current_selection_name = name ? g_strdup(name) : NULL;
     memcpy(&rfc->current_selection_format, fmt, sizeof(*fmt));
     if (rebuild) {
	  nosignal_flag = TRUE;
	  rebuild_strings(rfc);
	  nosignal_flag = FALSE;
     }
     g_signal_emit(G_OBJECT(rfc),
		     record_format_combo_signals[FORMAT_CHANGED_SIGNAL],0);
}

static void record_format_combo_selection_changed(Combo *obj)
{
     RecordFormatCombo *rfc = RECORD_FORMAT_COMBO(obj);
     int i,j;
     Dataformat *df;
     RecordFormat *rf;
     
     if (nosignal_flag) return;

     i = gtk_combo_box_get_active (GTK_COMBO_BOX (obj));
     if (i >= rfc->other_start) {
	  g_signal_emit(G_OBJECT(obj),
			  record_format_combo_signals
			  [FORMAT_DIALOG_REQUEST_SIGNAL],0);
	  rebuild_strings(rfc);
     } else if (i >= rfc->custom_start) { 
	  g_assert(i == rfc->custom_start && rfc->current_selection_type == 3);
	  /* Do nothing - custom already selected */
     } else if (i >= rfc->nameless_preset_start) {
	  j = i - rfc->nameless_preset_start;
	  df = list_object_get(rfc->nameless_presets, j);
	  record_format_set_main(rfc,2,NULL,df,TRUE);
     } else if (i >= rfc->named_preset_start) {
	  j = i - rfc->named_preset_start;
	  rf = list_object_get(rfc->named_presets, j);
	  g_assert(rf != NULL);
	  record_format_set_main(rfc,1,rf->name,&(rf->fmt),TRUE);
     } else {
	  /* Do nothing - none already selected */	  
     }
}

static void record_format_combo_class_init(RecordFormatComboClass *klass)
{
     GtkObjectClass *oc = GTK_OBJECT_CLASS(klass);
     oc->destroy = record_format_combo_destroy;
     COMBO_CLASS(klass)->selection_changed = record_format_combo_selection_changed;
     klass->format_changed = NULL;
     record_format_combo_signals[FORMAT_CHANGED_SIGNAL] = 
	  g_signal_new("format_changed", G_TYPE_FROM_CLASS(klass),
		       G_SIGNAL_RUN_LAST,
		       G_STRUCT_OFFSET(RecordFormatComboClass,
					   format_changed),
		       NULL, NULL,
		       g_cclosure_marshal_VOID__VOID,G_TYPE_NONE,0);
     record_format_combo_signals[FORMAT_DIALOG_REQUEST_SIGNAL] = 
	  g_signal_new("format_dialog_request", G_TYPE_FROM_CLASS(klass),
		       G_SIGNAL_RUN_LAST,
		       G_STRUCT_OFFSET(RecordFormatComboClass,
					   format_dialog_request),
		       NULL, NULL,
		       g_cclosure_marshal_VOID__VOID,G_TYPE_NONE,0);
}

static void record_format_combo_init(RecordFormatCombo *obj)
{
}

static void presets_changed(ListObject *lo, gpointer item, gpointer user_data)
{
     RecordFormatCombo *rfc = RECORD_FORMAT_COMBO(user_data);
     rebuild_strings(rfc);
}

GtkWidget *record_format_combo_new(ListObject *named_presets, 
				   ListObject *nameless_presets, 
				   gboolean show_other)
{
     RecordFormatCombo *rfc = g_object_new(RECORD_FORMAT_COMBO_TYPE, NULL);
     rfc->named_presets = named_presets;
     gtk_object_ref(GTK_OBJECT(named_presets));
     gtk_object_sink(GTK_OBJECT(named_presets));
     g_signal_connect_object(named_presets,"item_added",
				    G_CALLBACK(presets_changed),rfc,0);
     g_signal_connect_object(named_presets,"item_removed",
				    G_CALLBACK(presets_changed),rfc,0);
     g_signal_connect_object(named_presets,"item_notify",
				    G_CALLBACK(presets_changed),rfc,0);

     rfc->nameless_presets = nameless_presets;
     gtk_object_ref(GTK_OBJECT(nameless_presets));
     gtk_object_sink(GTK_OBJECT(nameless_presets));
     g_signal_connect_object(nameless_presets,"item_added",
				    G_CALLBACK(presets_changed),rfc,0);
     g_signal_connect_object(nameless_presets,"item_removed",
				    G_CALLBACK(presets_changed),rfc,0);
     g_signal_connect_object(nameless_presets,"item_notify",
				    G_CALLBACK(presets_changed),rfc,0);
     rfc->show_other = show_other;
     rfc->current_selection_type = 0;
     rebuild_strings(rfc);
     return GTK_WIDGET(rfc);
}

gboolean record_format_combo_set_named_preset(RecordFormatCombo *rfc, 
					     gchar *preset_name)
{
     GList *l;
     RecordFormat *rf;
     for (l=rfc->named_presets->list; l!=NULL; l=l->next) {
	  rf = (RecordFormat *)(l->data);
	  if (!strcmp(rf->name, preset_name)) break;
     }     
     if (l == NULL) return FALSE;

     record_format_set_main(rfc,1,rf->name,&rf->fmt,TRUE);
     return TRUE;
}

void record_format_combo_set_format(RecordFormatCombo *rfc,
				    Dataformat *fmt)
{
     record_format_set_main(rfc,2,NULL,fmt,TRUE);
}

void record_format_combo_store(RecordFormatCombo *rfc)
{
     if (rfc->stored_selection_name != NULL) 
	  g_free(rfc->stored_selection_name);
     rfc->stored_selection_type = rfc->current_selection_type;
     rfc->stored_selection_name = g_strdup(rfc->current_selection_name);
     memcpy(&rfc->stored_selection_format,&rfc->current_selection_format,
	    sizeof(Dataformat));
}

void record_format_combo_recall(RecordFormatCombo *rfc)
{
     record_format_set_main(rfc,rfc->stored_selection_type,
			    rfc->stored_selection_name,
			    &rfc->stored_selection_format,TRUE);
}

Dataformat *record_format_combo_get_format(RecordFormatCombo *rfc)
{
     return &rfc->current_selection_format;
}

gchar *record_format_combo_get_preset_name(RecordFormatCombo *rfc)
{
     if (rfc->current_selection_type == 1)
	  return rfc->current_selection_name;
     else
	  return NULL;
}
