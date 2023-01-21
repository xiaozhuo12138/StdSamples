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


#ifndef SANDWICHDIALOG_H_INCLUDED
#define SANDWICHDIALOG_H_INCLUDED

#include <gtk/gtk.h>
#include "effectdialog.h"
#include "documentlist.h"

#define SANDWICH_DIALOG_TYPE          (sandwich_dialog_get_type ())
#define SANDWICH_DIALOG(obj)          (G_TYPE_CHECK_INSTANCE_CAST ((obj), SANDWICH_DIALOG_TYPE, SandwichDialog))
#define IS_SANDWICH_DIALOG(obj)       (G_TYPE_CHECK_INSTANCE_TYPE ((obj), SANDWICH_DIALOG_TYPE))
#define SANDWICH_DIALOG_CLASS(klass)  (G_TYPE_CHECK_CLASS_CAST ((klass),  SANDWICH_DIALOG_TYPE, SandwichDialogClass))

typedef struct {
     EffectDialog ed;
     DocumentList *docsel;
     GtkToggleButton *align_begin,*align_end,*align_marker;
     GtkEntry *marker_entry;
} SandwichDialog;

typedef struct {
     EffectDialogClass edc;
} SandwichDialogClass;

GType sandwich_dialog_get_type(void);

#endif
