/*
 * Copyright (C) 2004 2005 2008 2009, Magnus Hjorth
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

#include <math.h>
#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>

#include "recorddialog.h"
#include "main.h"
#include "inifile.h"
#include "sound.h"
#include "um.h"
#include "int_box.h"
#include "formatselector.h"
#include "mainloop.h"

G_DEFINE_TYPE(RecordDialog,record_dialog,GTK_TYPE_WINDOW)

static struct {
     FormatSelector *fs;
     GtkEntry *name_entry;
     int old_choice;
     GtkWidget *preset_view;
     GtkButton *set_button_button;
} other_dialog;

static gboolean record_dialog_set_format(RecordDialog *rd);

static gboolean record_dialog_stopflag = FALSE;
static RecordDialog *current_dialog;

static ListObject *preset_list = NULL;
static GtkListStore *preset_liststore = NULL;

static void build_preset_list(void)
{
     gchar *s1,*s2,*s3,*s4,*s5,*s6;
     GList *l2 = NULL;
     int i;
     RecordFormat *rf;
     GtkTreeIter iter;

     if (preset_list != NULL) return;
     if (preset_liststore != NULL) return;
     preset_liststore = gtk_list_store_new(3,G_TYPE_STRING,G_TYPE_INT,G_TYPE_POINTER);

     /* Build format list */
     /* Support reading both old and new style formats */
     if (inifile_get("recordFormat1",NULL) == NULL &&
	 inifile_get("recordFormat1_Name",NULL) == NULL) {
	  inifile_set("recordFormat1","16/2/44100/S/CD Quality");
	  inifile_set("recordFormat2","8/1/8000//Low quality");
     }
     for (i=1; ; i++) {
	  s1 = g_strdup_printf("recordFormat%d",i);
	  s2 = inifile_get(s1,NULL);
	  g_free(s1);
	  if (s2 == NULL) {
	       s1 = g_strdup_printf("recordFormat%d_Name",i);
	       s2 = inifile_get(s1,NULL);
	       g_free(s1);
	       if (s2 == NULL) break;
	       rf = g_malloc(sizeof(*rf));
	       s3 = g_strdup_printf("recordFormat%d",i);
	       if (!dataformat_get_from_inifile(s3,TRUE,&(rf->fmt))) {
		    g_free(s3);
		    g_free(rf);
		    continue;
	       }
	       g_free(s3);
	       gtk_list_store_append(preset_liststore,&iter);
	       gtk_list_store_set(preset_liststore,&iter,0,s2,1,i,2,(gpointer)&rf->fmt,-1);
	       rf->num = i;
	       rf->name = g_strdup(s2);
	       l2 = g_list_append(l2,rf);
	  } else {
	       s3 = strchr(s2,'/');
	       if (s3 == NULL) continue;
	       s3 += 1;
	       s4 = strchr(s3,'/');
	       if (s4 == NULL) continue;
	       s4 += 1;
	       s5 = strchr(s4,'/');
	       if (s5 == NULL) continue;
	       s5 += 1;
	       s6 = strchr(s5,'/');
	       if (s6 == NULL) continue;
	       s6 += 1;
	       rf = g_malloc(sizeof(*rf));
	       rf->num = i;
	       rf->fmt.type = DATAFORMAT_PCM;
	       rf->fmt.samplesize = ((guint)strtod(s2,NULL)) / 8;
	       rf->fmt.channels = ((guint)strtod(s3,NULL));
	       rf->fmt.samplebytes = rf->fmt.samplesize * rf->fmt.channels;
	       rf->fmt.samplerate = ((guint)strtod(s4,NULL));
	       rf->fmt.sign = FALSE;
	       rf->fmt.bigendian = IS_BIGENDIAN;
	       rf->fmt.packing = 0;
	       for (; *s5 != 0; s5++) {
		    if (*s5 == 'B') rf->fmt.bigendian = !IS_BIGENDIAN; 
		    else if (*s5 == 'S') rf->fmt.sign = TRUE;
	       }
	       if (rf->fmt.samplesize < 1 || rf->fmt.samplesize > 4 ||
		   rf->fmt.channels < 1 || rf->fmt.channels > 8 ||
		   rf->fmt.samplerate < 1) {
		    g_free(rf);
		    continue;
	       }    
	       rf->name = g_strdup(s6);
	       gtk_list_store_append(preset_liststore,&iter);
	       gtk_list_store_set(preset_liststore,&iter,0,s6,1,i,2,(gpointer)&rf->fmt,-1);
	       l2 = g_list_append(l2,rf);
	  }
     }
     
     preset_list = list_object_new_from_list(l2,FALSE);
     gtk_object_ref(GTK_OBJECT(preset_list));
}

static void set_preset(gchar *name, Dataformat *fmt)
{
     RecordFormat *rf;
     int i;
     gchar *c,*d;
     gchar *cmp_name;
     gint num;
     Dataformat *cmp_fmt;
     GtkTreeModel *model;
     GtkTreeIter iter;
     gboolean valid;

     g_assert(name != NULL);

     /* If there is an old format with the same name, update it.
      * Otherwise, set the number to one higher than the
      * currently highest used number. */
     i = 0;
     rf = g_malloc(sizeof(*rf));
     rf->fmt = *fmt;
     model = GTK_TREE_MODEL(preset_liststore);
     valid = gtk_tree_model_get_iter_first(model,&iter);
     while (valid) {
          gtk_tree_model_get (model, &iter, 0, &cmp_name, 1, &num, 2, &cmp_fmt, -1);
          if (!strcmp(cmp_name,name)) {
               if (dataformat_equal(cmp_fmt,fmt)) return;
               gtk_list_store_set(preset_liststore,&iter,2,(gpointer)&rf->fmt,-1);
          }
          if (num > i) i = num;
          valid = gtk_tree_model_iter_next (model, &iter);
     }

     if (!valid) {
          gtk_list_store_append(preset_liststore,&iter);
          gtk_list_store_set(preset_liststore,&iter,0,name,1,i+1,2,(gpointer)&rf->fmt,-1);
     }
	  
     c = g_strdup_printf("recordFormat%d",i+1);
     d = g_strdup_printf("%s_Name",c);
     inifile_set(d,name);
     dataformat_save_to_inifile(c,fmt,TRUE);
     g_free(d);
     g_free(c);
}

static gboolean compare_preset(gchar *name, Dataformat *fmt)
{
     gchar *cmp_name;
     Dataformat *cmp_fmt;
     GtkTreeModel *model;
     GtkTreeIter iter;
     gboolean valid;
     model = GTK_TREE_MODEL(preset_liststore);
     valid = gtk_tree_model_get_iter_first(model,&iter);
     while (valid) {
          gtk_tree_model_get (model, &iter, 0, &cmp_name, 2, &cmp_fmt, -1);
          if (!strcmp(cmp_name,name))
               return dataformat_equal(cmp_fmt,fmt);
          valid = gtk_tree_model_iter_next (model, &iter);
     }
     return FALSE;
}

static void reset_peaks(GtkButton *button, gpointer user_data)
{
	int i;
	RecordDialog *rd = RECORD_DIALOG(user_data);
	for (i=0; i<8; i++) rd->maxpeak_values[i] = 0.0;
}

static void set_limit_label(RecordDialog *rd)
{
     gchar buf[64];
     if (rd->limit_record) {
	  get_time_s(rd->current_format->samplerate,
		     (rd->limit_bytes-rd->written_bytes) /
		     rd->current_format->samplebytes, 0, buf);
	  gtk_label_set_text(rd->limit_label,buf);	 
     } else
	  gtk_label_set_text(rd->limit_label,_("(no limit)"));
}

static gboolean process_input(RecordDialog *rd)
{
     guint32 x,y;
     gchar buf[4096]; 
     guint c,d,e;
     sample_t peak,rms,avg,*sp,*sq,s,pmax;
     guint32 clip_amount,clip_size;
     int i;
     gboolean finish_mode = record_dialog_stopflag;

     if (rd->current_format == NULL) return FALSE;
     /* Read input */
     input_store(rd->databuf);

     x = ringbuf_available(rd->databuf);
     /* Round to even samples */
     x -= x % rd->current_format->samplebytes;
     /* If we have <0.1 s of data, wait for more. */
     if ((x < rd->analysis_bytes && !finish_mode) || x==0) return FALSE;
     /* Send data older than 0.1s straight into the tempfile if we have one */
     if (!finish_mode)
	  x -= rd->analysis_bytes;
     while (x > 0) {
	  y = ringbuf_dequeue(rd->databuf,buf,MIN(x,sizeof(buf)));
	  if (rd->tf != NULL && !rd->paused) {
	       if (tempfile_write(rd->tf,buf,y)) {
		    record_dialog_stopflag = TRUE;
		    return FALSE;
	       }
	       rd->written_bytes += y;
	       if ( (rd->limit_record) && 
		    (rd->written_bytes >= rd->limit_bytes) ) {
		    record_dialog_stopflag = TRUE;
		    return TRUE;
	       }
	  }
	  x -= y;
     }
     if (finish_mode) return TRUE;
     /* Analyse data */
     y = ringbuf_dequeue(rd->databuf,rd->analysis_buf,rd->analysis_bytes);
     g_assert(y == rd->analysis_bytes);
     /* First write out the raw data to disk */
     if (rd->tf != NULL && !rd->paused) {
	  if (tempfile_write(rd->tf,rd->analysis_buf,y)) {
	       record_dialog_stopflag = TRUE;
	       return FALSE;
	  }
	  rd->written_bytes += y;
	  if ( (rd->limit_record) && (rd->written_bytes >= rd->limit_bytes) )
	       record_dialog_stopflag = TRUE;
     }
     convert_array(rd->analysis_buf,rd->current_format,
		   rd->analysis_sbuf,&dataformat_sample_t,
		   rd->analysis_samples, DITHER_NONE, NULL);
     /* Calculate statistics */
     sq = &(rd->analysis_sbuf[rd->analysis_samples]);
     d = rd->current_format->channels;
     for (c=0; c<d; c++) {
	  peak = rms = avg = 0.0;
	  clip_size = clip_amount = 0;
	  sp = &(rd->analysis_sbuf[c]);
	  for (; sp<sq; sp+=d) {
	       s = *sp;
	       avg += s;
	       s = fabs(s);
	       if (s > peak) peak = s;
	       rms += s*s;
	       *sp = s; /* Save the abs value for clipping calculation */
	  }
	  avg /= ((sample_t)rd->analysis_samples);
	  rms = sqrt((rms)/ ((sample_t)rd->analysis_samples) - avg*avg);
	  pmax = maximum_float_value(&(rd->current_format->fmt));
	  /* Since the conversion routines were changed for 1.2.9, this
	   * algo can give false alarms, but only when you're _very_ near 
	   * clipping. */
	  if (peak >= pmax) {
	       /* Calculate clipping amount and size */
	       sp = &(rd->analysis_sbuf[c]);
	       while (1) {
		    for (; sp<sq; sp+=d)
			 if (*sp >= pmax) break;
		    if (sp >= sq) break;
		    for (e=0; sp<sq && *sp>=pmax; e++,sp+=d) { }
		    clip_amount ++;
		    clip_size += e*e;
	       }	       
	  }
	  /* Set the labels and VU meters */
	  vu_meter_set_value(rd->meters[c],(float)peak);
	  g_snprintf(buf,sizeof(buf),"%.5f",(float)peak);
	  gtk_label_set_text(rd->peak_labels[c],buf);
	  if (((float)peak) > rd->maxpeak_values[c] )
	  {
	  	rd->maxpeak_values[c] = (float)peak;
		gtk_label_set_text(rd->maxpeak_labels[c],buf);
	  }
	  g_snprintf(buf,sizeof(buf),"%.5f",(float)rms);
	  gtk_label_set_text(rd->rms_labels[c],buf);
	  /* This probably needs some tweaking but has to do for now. */
	  if (clip_amount == 0) 
	       gtk_label_set_text(rd->clip_labels[c],_("None"));
	  else if (((float)clip_size)/((float)clip_amount) < 2.0
		   || (clip_size < rd->analysis_samples / (d * 100)))
	       gtk_label_set_text(rd->clip_labels[c],_("Mild"));
	  else 
	       gtk_label_set_text(rd->clip_labels[c],_("Heavy"));
     }
     if (rd->tf) {
	  get_time(rd->current_format->samplerate,
		   rd->written_bytes/rd->current_format->samplebytes,
		   0,buf,default_time_mode);
	  gtk_label_set_text(rd->time_label,buf);
	  g_snprintf(buf,200,"%" OFF_T_FORMAT "", 
		     (OFF_T_FTYPE)rd->written_bytes);
	  gtk_label_set_text(rd->bytes_label,buf);
	  get_time_s(rd->current_format->samplerate,
		     rd->written_bytes/rd->current_format->samplebytes,
		     0, buf);
	  if ( strcmp(buf, GTK_WINDOW(rd)->title) ) {
	       gtk_window_set_title(GTK_WINDOW(rd),buf);
	       /* Also update the remaining time here */
	       set_limit_label(rd);
	  }
	  i = input_overrun_count();
	  if (i>=0) {
	       i -= rd->overruns_before_start;
	       while ((guint) i > rd->overruns) {
		    if (rd->overruns < 10)
			 rd->overrun_locs[rd->overruns] = rd->written_bytes;
		    rd->overruns++;
	       }
	       g_snprintf(buf,sizeof(buf),"%d",i);
	       gtk_label_set_text(rd->overruns_label,buf);
	  }

     }
     return TRUE;
}

void input_ready_func(void)
{
     process_input(current_dialog);
}

static void record_dialog_format_changed(Combo *combo, gpointer user_data)
{
     RECORD_DIALOG(user_data)->format_changed = TRUE;
}

static gchar *other_dialog_get_name(void)
{
     gchar *c,*d;
     c = (gchar *)gtk_entry_get_text(other_dialog.name_entry);
     while (*c == ' ') c++; /* Skip beginning spaces */
     if (*c == 0) return NULL;
     c = g_strdup(c);
     /* Trim ending spaces */
     d = strchr(c,0);
     d--;
     while (*d == ' ') { *d = 0; d--; }
     return c;
}

static void other_dialog_ok (gpointer user_data)
{    
     RecordDialog *rd = RECORD_DIALOG(user_data);
     Dataformat df;
     gchar *c;

     if (format_selector_check(other_dialog.fs)) return;

     format_selector_get(other_dialog.fs,&df);

     c = other_dialog_get_name();
     if (c != NULL && compare_preset(c,&df))
	  record_format_combo_set_named_preset(rd->format_combo,c);
     else
	  record_format_combo_set_format(rd->format_combo,&df);
     g_free(c);
}

static void other_dialog_addpreset (gpointer user_data)
{
     gchar *name;
     Dataformat df;

     if (format_selector_check(other_dialog.fs)) return;
     format_selector_get(other_dialog.fs,&df);

     name = other_dialog_get_name();
     g_assert(name != NULL);
     
     set_preset(name,&df);
     /* b = record_format_combo_set_named_preset(rd->format_combo,name);
	g_assert(b); */
     g_free(name);
}

static void other_dialog_name_changed(GtkEditable *editable, 
				      gpointer user_data)
{
     gchar *c;
     GList *l;
     RecordFormat *rf;
     
     c = other_dialog_get_name();
     for (l=preset_list->list; l!=NULL; l=l->next) {
	  rf = (RecordFormat *)(l->data);
	  if (rf->name != NULL && c != NULL && !strcmp(rf->name,c))
	       break;
     }
     if (l != NULL) {
      gtk_button_set_label (other_dialog.set_button_button, _("Update preset"));
      gtk_widget_set_sensitive(GTK_WIDGET(other_dialog.set_button_button), TRUE);
     } else if (c != NULL) {
      gtk_button_set_label (other_dialog.set_button_button, _("Add preset"));
      gtk_widget_set_sensitive(GTK_WIDGET(other_dialog.set_button_button), TRUE);
     } else {
	  gtk_widget_set_sensitive(GTK_WIDGET(other_dialog.set_button_button),
				   FALSE);
     }
     g_free(c);
}

static void other_dialog_select_changed(GtkTreeSelection *sel,
				      gpointer user_data)
{
     GtkTreeModel *model;
     GtkTreeIter iter;
     gchar *name;
     Dataformat *fmt;
     if (!gtk_tree_selection_get_selected (sel, &model, &iter)) return;
     gtk_tree_model_get (model, &iter, 0, &name, 2, &fmt, -1);
     format_selector_set(other_dialog.fs, fmt);
     gtk_entry_set_text(other_dialog.name_entry, name);
}

static void
other_format_dlg_response (GtkDialog * dlg, int response, gpointer user_data)
{
   switch (response)
   {
      case GTK_RESPONSE_OK: /* set format */
         other_dialog_ok (user_data);
         g_signal_stop_emission_by_name (dlg, "response");
         return;
      case GTK_RESPONSE_APPLY: /* Add/Update preset */
         other_dialog_addpreset (user_data);
         g_signal_stop_emission_by_name (dlg, "response");
         return;
   }
   gtk_widget_destroy (GTK_WIDGET (dlg));
}

static void other_format_dialog(RecordFormatCombo *rfc, RecordDialog *rd)
{
     GtkWidget *b,*c,*d,*e,*f, * dialog;
     GtkTreeSelection *sel;
     GtkTreeViewColumn *col;
     GtkCellRenderer *renderer;

     dialog = gtk_dialog_new ();
     gtk_window_set_title (GTK_WINDOW (dialog), _("Custom format"));
     gtk_window_set_modal (GTK_WINDOW (dialog), TRUE);
     gtk_window_set_transient_for (GTK_WINDOW (dialog), GTK_WINDOW (rd));
     gtk_window_set_skip_pager_hint (GTK_WINDOW (dialog), TRUE);
     gtk_window_set_skip_taskbar_hint (GTK_WINDOW (dialog), TRUE);

     other_dialog.fs = FORMAT_SELECTOR(format_selector_new(TRUE));
     other_dialog.name_entry = GTK_ENTRY(gtk_entry_new());

     other_dialog.preset_view = gtk_tree_view_new();
     gtk_tree_view_set_headers_visible(GTK_TREE_VIEW(other_dialog.preset_view),FALSE);
     gtk_tree_view_set_model(GTK_TREE_VIEW(other_dialog.preset_view),GTK_TREE_MODEL(preset_liststore));
     renderer = gtk_cell_renderer_text_new();
     col = gtk_tree_view_column_new_with_attributes(NULL,renderer,"text",0,NULL);
     gtk_tree_view_append_column(GTK_TREE_VIEW(other_dialog.preset_view),col);
     sel = gtk_tree_view_get_selection(GTK_TREE_VIEW(other_dialog.preset_view));
     gtk_tree_selection_set_mode(sel,GTK_SELECTION_SINGLE);

     gtk_container_set_border_width (GTK_CONTAINER (dialog), 5);
     b = gtk_dialog_get_content_area (GTK_DIALOG (dialog));
     gtk_box_set_spacing (GTK_BOX (b), 5);

     c = gtk_hbox_new(FALSE,6);
     gtk_container_add(GTK_CONTAINER(b),c);
     d = gtk_vbox_new(FALSE,6);
     gtk_box_pack_start(GTK_BOX(c),d,FALSE,FALSE,0);
     e = GTK_WIDGET(other_dialog.fs);
     gtk_box_pack_start(GTK_BOX(d),e,FALSE,FALSE,0);
     e = gtk_hseparator_new();
     gtk_box_pack_start(GTK_BOX(d),e,FALSE,FALSE,0);
     e = gtk_label_new(_("The sign and endian-ness can usually be left at their "
		       "defaults, but should be changed if you're unable to "
		       "record or get bad sound."));
     gtk_label_set_line_wrap(GTK_LABEL(e),TRUE);
     gtk_box_pack_start(GTK_BOX(d),e,FALSE,FALSE,0);

     e = gtk_hbox_new(FALSE,4);
     gtk_box_pack_end(GTK_BOX(d),e,FALSE,FALSE,0);
     f = gtk_label_new(_("Name :"));
     gtk_container_add(GTK_CONTAINER(e),f);
     f = GTK_WIDGET(other_dialog.name_entry);
     gtk_container_add(GTK_CONTAINER(e),f);
     e = gtk_hseparator_new();
     gtk_box_pack_end(GTK_BOX(d),e,FALSE,FALSE,0);
     d = gtk_vseparator_new();
     gtk_box_pack_start(GTK_BOX(c),d,FALSE,FALSE,0);

     d = gtk_vbox_new(FALSE,6);
     gtk_container_add(GTK_CONTAINER(c),d);

     e = gtk_label_new(_("Presets:"));
     gtk_box_pack_start(GTK_BOX(d),e,FALSE,FALSE,0);

     e = gtk_scrolled_window_new(NULL,NULL);
     gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(e),GTK_POLICY_NEVER,
				    GTK_POLICY_AUTOMATIC);
     gtk_container_add(GTK_CONTAINER(d),e);

     f = GTK_WIDGET(other_dialog.preset_view);
     gtk_scrolled_window_add_with_viewport(GTK_SCROLLED_WINDOW(e),f);

     c = gtk_hseparator_new();
     gtk_box_pack_start(GTK_BOX(b),c,FALSE,FALSE,0);

     d = gtk_dialog_add_button (GTK_DIALOG (dialog), _("Set format"), GTK_RESPONSE_OK);
     d = gtk_dialog_add_button (GTK_DIALOG (dialog), _("Add preset"), GTK_RESPONSE_APPLY);
     gtk_widget_set_sensitive (d, FALSE);
     other_dialog.set_button_button = GTK_BUTTON (d);
     d = gtk_dialog_add_button (GTK_DIALOG (dialog), _("Close"), GTK_RESPONSE_CLOSE);

     g_signal_connect (dialog, "response",
                       G_CALLBACK (other_format_dlg_response), rd);
     gtk_widget_show_all (dialog);

     g_signal_connect(sel,"changed",
                      G_CALLBACK(other_dialog_select_changed),NULL);
     g_signal_connect(G_OBJECT(other_dialog.name_entry),"changed",
                      G_CALLBACK(other_dialog_name_changed),rd);
}

static void update_limit(RecordDialog *rd)
{
     gdouble d;
     if (rd->tf == NULL) return;
     d = (gdouble)rd->limit_seconds;
     d *= (gdouble)(rd->current_format->samplerate);
     d *= (gdouble)(rd->current_format->samplebytes);
     rd->limit_bytes = (off_t)d;
     set_limit_label(rd);
}

static void set_time_limit(GtkButton *button, gpointer user_data)
{
     gfloat f;
     gchar buf[64];
     RecordDialog *rd = RECORD_DIALOG(user_data);
     
     f = parse_time((gchar *)gtk_entry_get_text(rd->limit_entry));

     if (f < 0.0) {
	  popup_error(_("Invalid time value. Time must be specified in the form"
		      " (HH')MM:SS(.mmmm)"));
	  return;
     }
     
     rd->limit_record = TRUE;
     rd->limit_seconds = f;
     get_time_l(1000,(off_t)(f*1000.0),(off_t)(f*1000.0),buf);
     gtk_label_set_text(rd->limit_set_label, buf);
     gtk_widget_set_sensitive(GTK_WIDGET(rd->disable_limit_button),TRUE);
     update_limit(rd);
}

static void disable_time_limit(GtkButton *button, gpointer user_data)
{
     RecordDialog *rd = RECORD_DIALOG(user_data);
     rd->limit_record = FALSE;
     gtk_label_set_text(rd->limit_set_label,_("(no limit)"));
     gtk_widget_set_sensitive(GTK_WIDGET(rd->disable_limit_button),FALSE);     
}

static gboolean record_dialog_set_format(RecordDialog *rd)
{
     Dataformat *df;
     GtkWidget *a,*b;
     gint i;

     /* printf("record_dialog_set_format: %s fresh=%d\n",rf->name,
	rd->format_combo_fresh); */
     input_stop();
     rd->current_format = NULL;
     if (rd->vu_frame->child != NULL)
	  gtk_container_remove(GTK_CONTAINER(rd->vu_frame),
			       rd->vu_frame->child);
     g_free(rd->peak_labels);
     g_free(rd->maxpeak_labels);
     g_free(rd->rms_labels);
     g_free(rd->clip_labels);
     g_free(rd->meters);
     rd->peak_labels = NULL;
     rd->maxpeak_labels = NULL;
     rd->rms_labels = NULL;
     rd->clip_labels = NULL;
     rd->meters = NULL;
     gtk_label_set_text(rd->status_label,_("Format not selected"));
     gtk_widget_set_sensitive(rd->record_button,FALSE);
     gtk_widget_set_sensitive(rd->reset_button,FALSE);

     df = record_format_combo_get_format(rd->format_combo);
     i = input_select_format(df,FALSE,input_ready_func);
     if (i < 0) {
	  user_error(_("This format is not supported by the input driver!"));
	  return TRUE;
     } else if (i > 0)
	  return TRUE;

     gtk_label_set_text(rd->status_label,_("Ready for recording"));
     record_format_combo_store(rd->format_combo);
     rd->current_format = &(rd->format_combo->stored_selection_format);
     rd->peak_labels = g_malloc(df->channels*sizeof(GtkLabel *));
     rd->maxpeak_labels = g_malloc(df->channels*sizeof(GtkLabel *));
     rd->clip_labels = g_malloc(df->channels*sizeof(GtkLabel *));
     rd->rms_labels = g_malloc(df->channels*sizeof(GtkLabel *));
     rd->meters = g_malloc(df->channels*sizeof(VuMeter *));
     memset(rd->maxpeak_values,0,sizeof(rd->maxpeak_values));
     a = gtk_table_new(6*((df->channels+1)/2),4,FALSE);
     gtk_container_set_border_width(GTK_CONTAINER(a),4);
     
     for (i=0; i*2<df->channels; i++) {
	  attach_label(_("Peak: "),a,i*6+2,0);
	  attach_label(_("Peak max: "),a,i*6+3,0);
	  attach_label(_("RMS: "),a,i*6+4,0);
	  attach_label(_("Clipping: "),a,i*6+5,0);	  
     }
     for (i=0; i<df->channels; i++) {
	  b = gtk_label_new(channel_name(i,df->channels));
	  gtk_table_attach(GTK_TABLE(a),b,(i&1)+1,(i&1)+2,(i/2)*6+0,(i/2)*6+1,
			   0,0,0,0);
	  b = vu_meter_new(0.0);
	  rd->meters[i] = VU_METER(b);
	  gtk_table_attach(GTK_TABLE(a),b,(i&1)+1,(i&1)+2,(i/2)*6+1,(i/2)*6+2,
			   0,0,0,0);	  
	  rd->peak_labels[i] = attach_label("",a,(i/2)*6+2,(i&1)+1);
	  rd->maxpeak_labels[i] = attach_label("",a,(i/2)*6+3,(i&1)+1);
	  rd->rms_labels[i] = attach_label("",a,(i/2)*6+4,(i&1)+1);
	  rd->clip_labels[i] = attach_label(_("None"),a,(i/2)*6+5,(i&1)+1);
     }

     gtk_table_set_col_spacings(GTK_TABLE(a),5);
     gtk_table_set_row_spacings(GTK_TABLE(a),3);
     gtk_container_add(GTK_CONTAINER(rd->vu_frame),a);
     gtk_widget_show_all(a);

     gtk_widget_set_sensitive(rd->record_button,TRUE);
     gtk_widget_set_sensitive(rd->reset_button,TRUE);

     /* Create a 2 second ring buffer */
     if (rd->databuf != NULL) {
	  ringbuf_free(rd->databuf);
	  g_free(rd->analysis_buf);
	  g_free(rd->analysis_sbuf);
     }
     rd->databuf = ringbuf_new(2*df->samplebytes*df->samplerate);
     /* Do analysis on 0.1 s parts. */
     rd->analysis_bytes = df->samplebytes * df->samplerate / 10;
     rd->analysis_samples = df->channels * df->samplerate / 10;
     rd->analysis_buf = g_malloc(rd->analysis_bytes);
     rd->analysis_sbuf = g_malloc(sizeof(sample_t) * rd->analysis_samples); 

     /* Call process_input manually one time here
      * (required for some sound drivers to start recording) */
     process_input(rd);

     return FALSE;
}

static void check_format_change(RecordDialog *rd)
{
     if (!rd->format_changed)
	  return;

     rd->format_changed = FALSE;
     record_dialog_set_format(rd);
}

static void record_dialog_start(GtkButton *button, gpointer user_data)
{
     RecordDialog *rd = RECORD_DIALOG(user_data);
     GtkRequisition req;
     int i;
     gchar *c;

     if (rd->tf != NULL) {
	  /* This is a hack to prevent the window from resizing when we 
	   * change the button's caption */
	  gtk_widget_size_request(rd->record_button,&req);
	  gtk_widget_set_size_request(rd->record_button,req.width,req.height);

	  /* Toggle pause mode */
	  rd->paused = !rd->paused;	  
	  gtk_label_set_text(GTK_LABEL(GTK_BIN(rd->record_button)->child),
			     rd->paused?_("Resume recording"):_("Pause recording"));
	  gtk_label_set_text(rd->status_label,rd->paused?
			     translate_strip(N_("RecordStatus|Paused")):
			     translate_strip(N_("RecordStatus|Recording")));
	  return;
     }

     inifile_set_gboolean("limitRecord",rd->limit_record);
     inifile_set_gfloat( "limitSecs",rd->limit_seconds);

     c = record_format_combo_get_preset_name(rd->format_combo);
     inifile_set("lastRecordFormat",c);
     if (c == NULL) {
	  dataformat_save_to_inifile("lastRecordFormat",
				     record_format_combo_get_format
				     (rd->format_combo), TRUE);
     }

     gtk_widget_set_sensitive(GTK_WIDGET(rd->format_combo),FALSE);
     gtk_label_set_text(GTK_LABEL(GTK_BIN(rd->record_button)->child),
			_("Pause recording"));
     rd->paused = FALSE;
     gtk_label_set_text(GTK_LABEL(GTK_BIN(rd->close_button)->child),
			_("Finish"));
     rd->tf = tempfile_init(rd->current_format,TRUE);
     rd->written_bytes = 0;
     i = input_overrun_count();
     gtk_label_set_text(rd->status_label,
			translate_strip(N_("RecordStatus|Recording")));
     if (i>-1) {
	  rd->overruns_before_start = i;
	  gtk_label_set_text(rd->overruns_title,_("Overruns: "));
     }
     rd->overruns = 0;
     gtk_label_set_text(rd->bytes_text_label,_("Bytes written: "));
     gtk_label_set_text(rd->limit_text_label,_("Auto stop in: "));
     update_limit(rd);
}

static void record_dialog_close(GtkButton *button, gpointer user_data)
{
     record_dialog_stopflag = TRUE;
}

static gboolean record_dialog_delete_event(GtkWidget *widget, GdkEvent *event,
					   gpointer user_data)
{
     record_dialog_stopflag = TRUE;
     return TRUE;
}

void record_dialog_init(RecordDialog *obj)
{
     GtkWidget *a,*b,*c,*d,*e;
     GtkAccelGroup* ag;
     GtkRequisition req;  
     gchar limitbuf[64];
     gchar *s1;
     GList *dp;
     gboolean complete;
     Dataformat df,*dfp;

     ag = gtk_accel_group_new();

     obj->format_changed = FALSE;
     obj->current_format = NULL;
     obj->databuf = NULL;
     obj->meters = NULL;
     obj->peak_labels = obj->maxpeak_labels = obj->clip_labels = NULL;
     obj->rms_labels = NULL;
     obj->tf = NULL;
     obj->written_bytes = 0;
     obj->analysis_buf = NULL;
     obj->analysis_sbuf = NULL;
     obj->limit_record = inifile_get_gboolean("limitRecord",FALSE);
     obj->limit_seconds = inifile_get_gfloat("limitSecs",3600.0);
     if (obj->limit_seconds < 0.0) obj->limit_seconds = 3600.0;
     get_time_l(1000,(off_t)(obj->limit_seconds*1000.0),0,limitbuf);

     gtk_window_set_title(GTK_WINDOW(obj),_("Record"));
     gtk_window_set_modal(GTK_WINDOW(obj),TRUE);
     gtk_window_set_default_size(GTK_WINDOW(obj),320,400);
     gtk_window_set_position(GTK_WINDOW(obj),GTK_WIN_POS_CENTER);
     gtk_container_set_border_width(GTK_CONTAINER(obj),10);

     g_signal_connect(G_OBJECT(obj),"delete_event",
			G_CALLBACK(record_dialog_delete_event),NULL);



     build_preset_list();
     dp = input_supported_formats(&complete);
     obj->driver_presets = list_object_new_from_list(dp,FALSE);
     gtk_object_ref(GTK_OBJECT(obj->driver_presets));
     gtk_object_sink(GTK_OBJECT(obj->driver_presets));

     /* Add components */
     a = gtk_vbox_new(FALSE,10);
     gtk_container_add(GTK_CONTAINER(obj),a);
     b = gtk_frame_new(_("Recording settings"));
     gtk_box_pack_start(GTK_BOX(a),b,FALSE,FALSE,0);
     c = gtk_vbox_new(FALSE,0);
     gtk_container_set_border_width(GTK_CONTAINER(c),5);
     gtk_container_add(GTK_CONTAINER(b),c);
     d = gtk_hbox_new(FALSE,0);
     gtk_box_pack_start(GTK_BOX(c),d,FALSE,FALSE,0);
     e = gtk_label_new(_("Sample format: "));
     gtk_box_pack_start(GTK_BOX(d),e,FALSE,FALSE,0);

     e = record_format_combo_new(complete ? list_object_new(FALSE):preset_list,
				 obj->driver_presets, !complete);
     gtk_box_pack_start(GTK_BOX(d),e,TRUE,TRUE,0);
     obj->format_combo = RECORD_FORMAT_COMBO(e);
     g_signal_connect(G_OBJECT(e),"format_changed",
			G_CALLBACK(record_dialog_format_changed),obj);
     g_signal_connect(G_OBJECT(e),"format_dialog_request",
			G_CALLBACK(other_format_dialog),obj);

     d = gtk_hbox_new(FALSE,3);
     gtk_box_pack_start(GTK_BOX(c),d,FALSE,FALSE,6);

     e = gtk_label_new(_("Time limit: "));
     gtk_box_pack_start(GTK_BOX(d),e,FALSE,FALSE,0);
     e = gtk_label_new(_("(no limit)"));
     obj->limit_set_label = GTK_LABEL(e);
     if (obj->limit_record)
	  gtk_label_set_text(obj->limit_set_label, limitbuf);
     gtk_box_pack_start(GTK_BOX(d),e,FALSE,FALSE,0);

     e = gtk_button_new_with_label(_("Disable"));
     obj->disable_limit_button = GTK_BUTTON(e);
     gtk_widget_set_sensitive(GTK_WIDGET(e),obj->limit_record);
     g_signal_connect(G_OBJECT(e),"clicked",
			G_CALLBACK(disable_time_limit),obj);
     gtk_box_pack_end(GTK_BOX(d),e,FALSE,FALSE,0);
     e = gtk_button_new_with_label(_("Set"));
     g_signal_connect(G_OBJECT(e),"clicked",
			G_CALLBACK(set_time_limit),obj);
     gtk_box_pack_end(GTK_BOX(d),e,FALSE,FALSE,0);
     e = gtk_entry_new();
     obj->limit_entry = GTK_ENTRY(e);
     gtk_entry_set_text(obj->limit_entry, limitbuf );
     /* Max length = "hhh'mm:ss.mmm" */
     gtk_entry_set_max_length( obj->limit_entry, 14); 
     gtk_box_pack_end(GTK_BOX(d),e,FALSE,FALSE,0);     


     b = gtk_frame_new(_("Input levels"));
     obj->vu_frame = GTK_BIN(b);
     gtk_box_pack_start(GTK_BOX(a),b,TRUE,TRUE,0);
     b = gtk_table_new(3,4,FALSE);
     gtk_box_pack_start(GTK_BOX(a),b,FALSE,FALSE,0);
     attach_label(_("Recording status: "),b,0,0);
     obj->status_label = GTK_LABEL(gtk_label_new(_("Format not selected")));
     gtk_table_attach(GTK_TABLE(b),GTK_WIDGET(obj->status_label),1,4,0,1,
		      GTK_FILL,0,0,0);
     gtk_misc_set_alignment(GTK_MISC(obj->status_label),0.0,0.5);
     attach_label(_("Time recorded: "),b,1,0);
     obj->time_label = attach_label(_("N/A"),b,1,1);
     gtk_widget_size_request(GTK_WIDGET(obj->time_label),&req);
     /* Stops wobble during recording */
     gtk_widget_set_size_request(GTK_WIDGET(obj->time_label),150,req.height);

     obj->limit_text_label = attach_label("",b,2,0);
     obj->limit_label = attach_label("",b,2,1);
     obj->bytes_text_label = attach_label("",b,1,2);
     obj->bytes_label = attach_label("",b,1,3);     
     obj->overruns_title = attach_label("",b,2,2);
     obj->overruns_label = attach_label("",b,2,3);
     b = gtk_hbutton_box_new();
     gtk_button_box_set_layout(GTK_BUTTON_BOX(b),GTK_BUTTONBOX_END);
     gtk_box_pack_start(GTK_BOX(a),b,FALSE,FALSE,0);
     c = gtk_button_new_with_label(_("Start recording"));
     gtk_widget_add_accelerator (c, "clicked", ag, GDK_KEY(KP_Enter), 0, 
				 (GtkAccelFlags) 0);
     gtk_widget_add_accelerator (c, "clicked", ag, GDK_KEY(Return), 0, 
				 (GtkAccelFlags) 0);
     gtk_widget_set_sensitive(c,FALSE);
     g_signal_connect(G_OBJECT(c),"clicked",
			G_CALLBACK(record_dialog_start),obj);
     obj->record_button = GTK_WIDGET(c);
     gtk_container_add(GTK_CONTAINER(b),c);

     c = gtk_button_new_with_label(_("Reset max peaks"));
     gtk_widget_set_sensitive(c,FALSE);
     g_signal_connect(G_OBJECT(c),"clicked", G_CALLBACK(reset_peaks),obj);
     obj->reset_button = GTK_WIDGET(c);
     gtk_container_add(GTK_CONTAINER(b),c);     

     c = gtk_button_new_with_label(_("Launch mixer"));
     g_signal_connect(G_OBJECT(c),"clicked",G_CALLBACK(launch_mixer),
			NULL);
     gtk_container_add(GTK_CONTAINER(b),c);
     c = gtk_button_new_with_label(_("Close"));
     gtk_widget_add_accelerator (c, "clicked", ag, GDK_KEY(Escape), 0, 
				 (GtkAccelFlags) 0);
     g_signal_connect(G_OBJECT(c),"clicked",
			G_CALLBACK(record_dialog_close),obj);
     gtk_container_add(GTK_CONTAINER(b),c);
     obj->close_button = c;

     gtk_widget_show_all(a);
     gtk_window_add_accel_group(GTK_WINDOW (obj), ag);

     /* Special case: Only one format supported. Choose that format and 
      * make the format combo insensitive */
     if (complete && list_object_get_size(obj->driver_presets)==1) {
	  dfp = (Dataformat *)list_object_get(obj->driver_presets,0);
	  record_format_combo_set_format(obj->format_combo,dfp);
	  gtk_widget_set_sensitive(GTK_WIDGET(obj->format_combo),FALSE);
	  return;
     } 

     /* Set the last used format */
     s1 = inifile_get("lastRecordFormat",NULL);
     if (s1 != NULL) {	  
	  record_format_combo_set_named_preset(obj->format_combo,s1);
     } else {
	  if (dataformat_get_from_inifile("lastRecordFormat",TRUE,&df))
	       record_format_combo_set_format(obj->format_combo,&df);
     }
}

static void record_dialog_destroy(GtkObject *obj)
{
     RecordDialog *rd = RECORD_DIALOG(obj);
     if (rd->databuf) ringbuf_free(rd->databuf);
     rd->databuf = NULL;
     g_free(rd->analysis_buf);
     rd->analysis_buf = NULL;
     g_free(rd->analysis_sbuf);
     rd->analysis_sbuf = NULL;
     GTK_OBJECT_CLASS(record_dialog_parent_class)->destroy(obj);
     if (rd->driver_presets != NULL) {
	  list_object_foreach(rd->driver_presets, (GFunc)g_free, NULL);
	  list_object_clear(rd->driver_presets, FALSE);
	  gtk_object_unref(GTK_OBJECT(rd->driver_presets));
	  rd->driver_presets = NULL;
     }
}

static void record_dialog_class_init(RecordDialogClass *klass)
{
     GtkObjectClass *oc = GTK_OBJECT_CLASS(klass);
     oc->destroy = record_dialog_destroy;
}

Chunk *record_dialog_execute(int *noverruns, off_t overrun_locs[10])
{
     RecordDialog *rd;
     Chunk *ds;
     int i;

     rd = g_object_new(RECORD_DIALOG_TYPE, NULL);
     record_dialog_stopflag = FALSE;
     current_dialog = rd;
     gtk_widget_show(GTK_WIDGET(rd));
     while (!record_dialog_stopflag) {
	  mainloop();
	  check_format_change(rd); 
     }
     if (rd->tf != NULL) {
	  input_stop_hint();
	  i = 0; /* Just to avoid infinite loops */
	  while (process_input(rd) && i<128) { i++; }
	  input_stop();
	  ds = tempfile_finished(rd->tf);
	  *noverruns = rd->overruns;
	  memcpy(overrun_locs,rd->overrun_locs,10*sizeof(off_t));
	  gtk_widget_destroy(GTK_WIDGET(rd));
	  return ds;
     } else {
	  input_stop();
	  gtk_widget_destroy(GTK_WIDGET(rd));
	  return NULL;
     }
}
