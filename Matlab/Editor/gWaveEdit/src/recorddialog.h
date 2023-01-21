/*
 * Copyright (C) 2002 2003 2004 2009, Magnus Hjorth
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


#ifndef RECORDDIALOG_H_INCLUDED
#define RECORDDIALOG_H_INCLUDED

#include <gtk/gtk.h>
#include "chunk.h"
#include "int_box.h"
#include "datasource.h"
#include "vu_meter.h"
#include "ringbuf.h"
#include "tempfile.h"
#include "combo.h"
#include "recordformatcombo.h"

#define RECORD_DIALOG_TYPE          (record_dialog_get_type ())
#define RECORD_DIALOG(obj)          (G_TYPE_CHECK_INSTANCE_CAST ((obj), RECORD_DIALOG_TYPE, RecordDialog))
#define IS_RECORD_DIALOG(obj)       (G_TYPE_CHECK_INSTANCE_TYPE ((obj), RECORD_DIALOG_TYPE))
#define RECORD_DIALOG_CLASS(klass)  (G_TYPE_CHECK_CLASS_CAST ((klass),  RECORD_DIALOG_TYPE, RecordDialogClass))


typedef struct {

     GtkWindow window;

     RecordFormatCombo *format_combo;
     gboolean format_changed;
     GtkBin *vu_frame;
     GtkLabel *status_label,*time_label,*overruns_label,*overruns_title;
     GtkLabel *bytes_label,*bytes_text_label,*limit_label,*limit_text_label;
     GtkWidget *record_button,*reset_button,*close_button;     

     Dataformat *current_format;
     ListObject *driver_presets;
     gboolean limit_record;
     GtkEntry *limit_entry;
     GtkLabel *limit_set_label;
     GtkButton *disable_limit_button;
     gfloat limit_seconds;
     float maxpeak_values[8];
     GtkLabel **peak_labels,**maxpeak_labels,**clip_labels,**rms_labels;
     VuMeter **meters;

     TempFile *tf;
     gboolean paused;
     off_t written_bytes;
     off_t limit_bytes;
     Ringbuf *databuf;
     guint32 analysis_bytes;
     guint32 analysis_samples;
     gchar *analysis_buf;
     sample_t *analysis_sbuf;
     guint overruns_before_start, overruns;
     off_t overrun_locs[10];
} RecordDialog;

typedef struct {
     GtkWindowClass wc;
} RecordDialogClass;

GType record_dialog_get_type(void);
Chunk *record_dialog_execute(int *noverruns, off_t overrun_locs[10]);

#endif
