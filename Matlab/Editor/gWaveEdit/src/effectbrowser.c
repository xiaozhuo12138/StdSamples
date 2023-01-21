/*
 * Copyright (C) 2003 2004 2007 2009 2011, Magnus Hjorth
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
#include <gdk/gdkkeysyms.h>

#include <string.h>
#include "effectbrowser.h"
#include "effectdlg_volume.h"
#include "effectdlg_speed.h"
#include "effectdlg_samplerate.h"
#include "effectdlg_samplesize.h"
#include "effectdlg_combinechannels.h"
#include "effectdlg_pipe.h"
#include "inifile.h"
#include "documentlist.h"
#include "um.h"
#include "effectdlg_mapchannels.h"
#include "effectdlg_sandwich.h"

G_DEFINE_TYPE(EffectBrowser,effect_browser,GTK_TYPE_WINDOW)

struct source {
     gchar tag;
     gchar *name;
     effect_register_rebuild_func rebuild_func;
     effect_register_get_func get_func;
     gpointer rebuild_func_data,get_func_data;
     int is_new;
};

struct effect {
     gchar *name,*title,*location,*author,source_tag;
     gboolean process_tag;
};

static ListObject *effect_list = NULL;
static GSList *sources = NULL;

static GSList *geometry_stack = NULL;
static gboolean geometry_stack_inited = FALSE;

static void list_widget_rebuild(gpointer dummy, gpointer dummy2, 
				EffectBrowser *eb);

void effect_register_add_source(gchar *name, gchar tag,
				effect_register_rebuild_func rebuild_func,
				gpointer rebuild_func_data,
				effect_register_get_func get_func,
				gpointer get_func_data)
{
     struct source *s;
     s = g_malloc(sizeof(*s));
     s->tag = tag;
     s->name = name;
     s->rebuild_func = rebuild_func;
     s->rebuild_func_data = rebuild_func_data;
     s->get_func = get_func;
     s->get_func_data = get_func_data;
     s->is_new = TRUE;
     sources = g_slist_append(sources,s);
}

void effect_register_add_effect(gchar source_tag, const gchar *name, 
				const gchar *title, const gchar *author, 
				const gchar *location)
{
     struct effect *e;
     e = g_malloc(sizeof(*e));
     e->source_tag = source_tag;
     e->name = g_strdup(name);
     e->title = g_strdup(title);
     e->author = g_strdup(author);
     e->location = g_strdup(location);
     list_object_add(effect_list, e);
}

static void builtin_rebuild_func(gchar source_tag, gpointer user_data)
{
     gchar *author = _("Built-in");
     static const gchar loc[] = "";     

     effect_register_add_effect(source_tag,"volume",_("Volume adjust/fade"),
				author,loc);
     effect_register_add_effect(source_tag,"srate",_("Convert samplerate"),
				author,loc);
     effect_register_add_effect(source_tag,"ssize",_("Convert sample format"),
				author,loc);
     effect_register_add_effect(source_tag,"mapchannels",_("Map channels"),
				author,loc);
     effect_register_add_effect(source_tag,"combine",_("Combine channels"),
				author,loc);
     effect_register_add_effect(source_tag,"sandwich",
				_("Add channels from other file"),author,loc);
     effect_register_add_effect(source_tag,"speed",_("Speed"),author,loc);
     effect_register_add_effect(source_tag,"pipe",_("Pipe through program"),
				author,loc);
}

static EffectDialog *builtin_get_func(gchar *name, gchar source_tag,
				      gpointer user_data)
{
     GType type = 0;
     if (!strcmp(name,"volume")) type = volume_dialog_get_type();
     else if (!strcmp(name,"srate")) type = samplerate_dialog_get_type();
     else if (!strcmp(name,"ssize")) type = samplesize_dialog_get_type();
     else if (!strcmp(name,"mapchannels")) 
	  type = map_channels_dialog_get_type();
     else if (!strcmp(name,"combine")) 
	  type = combine_channels_dialog_get_type();
     else if (!strcmp(name,"speed")) type = speed_dialog_get_type();
     else if (!strcmp(name,"pipe")) type = pipe_dialog_get_type();
     else if (!strcmp(name,"sandwich")) type = sandwich_dialog_get_type();
     if (type != 0) 
	  return EFFECT_DIALOG(g_object_new(type, NULL));
     else
	  return NULL;
}

void effect_register_init(void)
{
     /* Add built-in effects source */
     effect_register_add_source("Built-in",'B',builtin_rebuild_func,NULL,
				builtin_get_func,NULL);
}

static void effect_register_update_list(void)
{
     GSList *s;
     struct source *src;
     gboolean b = FALSE;
     
     if (effect_list == NULL)
	  effect_list = list_object_new(FALSE);

     for (s=sources; s!=NULL; s=s->next) {
	  src = (struct source *)s->data;
	  if (src -> is_new) {
	       /* TODO: Cache instead of requesting from source each time */
	       src->rebuild_func(src->tag, src->rebuild_func_data);
	       src->is_new = FALSE;
	       b = TRUE;
	  }
     }
     
     if (b) list_object_notify(effect_list,NULL);
}

void free_effect(struct effect *e)
{
     g_free(e->name);
     g_free(e->title);
     g_free(e->location);
     g_free(e->author);
     g_free(e);
}

void effect_register_rebuild(void)
{
     GSList *s;
     struct source *src;
     list_object_foreach(effect_list,(GFunc)free_effect,NULL);
     list_object_clear(effect_list,FALSE);
     for (s=sources; s!=NULL; s=s->next) {
	  src = (struct source *)s->data;
	  src->is_new = TRUE;
     }
     effect_register_update_list();
}

static void effect_browser_remove_effect(EffectBrowser *eb)
{
     if (eb->current_dialog >= 0) 
	  gtk_container_remove
	       (GTK_CONTAINER(eb->dialog_container),
		GTK_WIDGET(eb->dialogs[eb->current_dialog]));
     eb->current_dialog = -1;
}

static void effect_browser_destroy (GObject *obj)
{
     // this is triggered by gtk_widget_destroy()
     EffectBrowser *eb = EFFECT_BROWSER(obj);
     guint i;
     effect_browser_remove_effect(eb);
     for (i=0; i<EFFECT_BROWSER_CACHE_SIZE; i++) {
	  if (eb->dialogs[i] != NULL) {
	       g_object_unref(GTK_WIDGET(eb->dialogs[i]));
	       eb->dialogs[i] = NULL;
	  }
     }
     G_OBJECT_CLASS(effect_browser_parent_class)->dispose(obj);
}

static void geom_push(EffectBrowser *eb)
{
     gchar *c;
     guint pos;
     /* This seems to be the only way to find out handle position */
     pos = GTK_WIDGET(eb->effect_list_container)->allocation.width;
     c = g_strdup_printf("%d",pos);
     geometry_stack_push(GTK_WINDOW(eb),c,&geometry_stack);
     g_free(c);
}

static gint effect_browser_delete_event(GtkWidget *widget, GdkEventAny *event)
{
     geom_push(EFFECT_BROWSER(widget));
     if (GTK_WIDGET_CLASS(effect_browser_parent_class)->delete_event)
	  return GTK_WIDGET_CLASS(effect_browser_parent_class)->delete_event(widget,event);
     else
	  return FALSE;
}

static void effect_browser_class_init(EffectBrowserClass *klass)
{
     GObjectClass *oc = G_OBJECT_CLASS(klass);
     GtkWidgetClass *wc = GTK_WIDGET_CLASS(klass);
     oc->dispose = effect_browser_destroy;
     wc->delete_event = effect_browser_delete_event;
}

static void effect_browser_close(EffectBrowser *eb)
{
     geom_push(eb);
     gtk_widget_destroy(GTK_WIDGET(eb));
}

static void apply_click(GtkWidget *widget, EffectBrowser *eb)
{
     gboolean ca,b,mwf;
     if (eb->dl->selected == NULL) {
	  user_error(_("You have no open file to apply the effect to!"));
	  return;
     }

     ca = gtk_toggle_button_get_active(eb->close_after);
     mwf = inifile_get_gboolean("mainwinFront",TRUE);

     if (ca) gtk_widget_hide(GTK_WIDGET(eb));     
     
     b = effect_dialog_apply(eb->dialogs[eb->current_dialog]);
     
     if (ca) {
	  if (b) 
	       gtk_widget_show(GTK_WIDGET(eb));
	  else
	       effect_browser_close(eb);
     } else if (!mwf)
	  gdk_window_raise(GTK_WIDGET(eb)->window);
}

static EffectDialog *get_effect_missing_dialog(gchar *name, gchar source_tag)
{
     EffectDialog *ed;
     GtkWidget *w;
     ed = g_object_new(EFFECT_DIALOG_TYPE, NULL);
     w = gtk_label_new(_("This effect could not be loaded."));
     gtk_container_add(ed->input_area,w);
     gtk_widget_show(w);
     return ed;
}

static void effect_browser_set_effect_main(EffectBrowser *eb, struct effect *e)
{
     int i;
     EffectDialog *ed;
     GSList *s;
     struct source *src;
     gchar *c;

     effect_browser_remove_effect(eb);

     /* Check dialog cache */
     for (i=0; i<EFFECT_BROWSER_CACHE_SIZE; i++) {
	  if (eb->dialog_effects[i] == e) break;
     }

     if (i >= EFFECT_BROWSER_CACHE_SIZE) {
	  /* Dialog not in cache */

	  /* Make room in top of cache */
	  for (i=0; i<EFFECT_BROWSER_CACHE_SIZE; i++) {
	       if (eb->dialog_effects[i] == NULL) break;
	  }
	  if (i >= EFFECT_BROWSER_CACHE_SIZE) {
	       /* No room in cache, throw out last element */
	       i = EFFECT_BROWSER_CACHE_SIZE-1;
	       gtk_object_unref(GTK_OBJECT(eb->dialogs[i]));
	       eb->dialogs[i] = NULL;
	       eb->dialog_effects[i] = NULL;
	  }
	  for (; i>0; i--) {
	       eb->dialogs[i] = eb->dialogs[i-1];
	       eb->dialog_effects[i] = eb->dialog_effects[i-1];
	  }

	  /* Get the new dialog */

	  ed = NULL;
	  for (s=sources; s!=NULL; s=s->next) {
	       src = (struct source *)s->data;
	       if (src->tag == e->source_tag) {
		    ed = src->get_func(e->name, e->source_tag,
				       src->get_func_data);
		    effect_dialog_setup(ed, e->name, eb);
		    break;
	       }
	  }

	  if (ed == NULL)
	       ed = get_effect_missing_dialog(e->name,e->source_tag);	  

	  g_assert(i == 0);
	  eb->dialogs[i] = ed;
	  gtk_object_ref(GTK_OBJECT(ed));
	  gtk_object_sink(GTK_OBJECT(ed));
	  eb->dialog_effects[i] = e;
     }

     eb->current_dialog = i;

     gtk_container_add(eb->dialog_container,
		       GTK_WIDGET(eb->dialogs[i]));
     gtk_widget_show(GTK_WIDGET(eb->dialogs[i]));

     c = g_strdup_printf("%c%s",e->source_tag,e->name);
     inifile_set("lastEffect",c);
     g_free(c);
}

void effect_browser_invalidate_effect(EffectBrowser *eb, gchar *effect_name, 
				      gchar source_tag)
{
     gboolean displayed = FALSE;
     struct effect *e;
     gint i=0;

     /* Search the cache for the effect */
     for (i=0; i<EFFECT_BROWSER_CACHE_SIZE; i++) {
	  e = eb->dialog_effects[i];
	  if (e != NULL && e->source_tag == source_tag && 
	      !strcmp(e->name, effect_name))
	       break;
     }

     if (i >= EFFECT_BROWSER_CACHE_SIZE) return; /* Not found */
     
     displayed = (i == eb->current_dialog);
     if (displayed) effect_browser_remove_effect(eb);
     gtk_object_unref(GTK_OBJECT(eb->dialogs[i]));
     eb->dialogs[i] = NULL;
     eb->dialog_effects[i] = NULL;
     if (displayed) effect_browser_set_effect_main(eb,e);
}

static void effect_browser_selection_changed(GtkTreeSelection *sel,
					gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);     
     struct effect *effect;
     GtkTreeModel *model;
     GtkTreeIter iter;
     if (gtk_tree_selection_get_selected (sel, &model, &iter)) {
          gtk_tree_model_get (model, &iter, 1, &effect, -1);
          effect_browser_set_effect_main(eb,effect);
     }
}

static void save_effect_order(EffectBrowser *eb)
{
     gint i;
     gchar *c,*d;
     struct effect *effect;
     gboolean valid;
     GtkTreeModel *model;
     GtkTreeIter iter;
     model = GTK_TREE_MODEL(eb->liststore);
     valid = gtk_tree_model_get_iter_first(model,&iter);
     i = 0;
     while(valid) {
          gtk_tree_model_get (model, &iter, 1, &effect, -1);
	  c = g_strdup_printf("effectBrowserOrder%d",i);
	  d = g_strdup_printf("%c%s",effect->source_tag,effect->name);
	  inifile_set(c,d);
	  g_free(c);
	  g_free(d);
          valid = gtk_tree_model_iter_next (model, &iter);
          i++;
     }
     c = g_strdup_printf("effectBrowserOrder%d",i);
     inifile_set(c,NULL);
     g_free(c);
}

static void move_up_down_main(EffectBrowser *eb, gboolean up)
{
     GtkTreeModel *model;
     GtkTreeIter iter1, iter2;
     GtkTreePath *path1, *path2;
     gboolean success = FALSE;
     if (gtk_tree_selection_get_selected (eb->list_sel, &model, &iter1)) {
          path1 = gtk_tree_model_get_path (model, &iter1);
          path2 = gtk_tree_path_copy (path1);
          if (up)
	       gtk_tree_path_prev (path2);
          else
	       gtk_tree_path_next (path2);
          if (gtk_tree_path_compare (path1,path2)) {
               if (gtk_tree_model_get_iter (model, &iter2, path2)) {
                    gtk_list_store_swap (eb->liststore, &iter1, &iter2);
                    success = TRUE;
               }
          }
          gtk_tree_path_free(path1);
          gtk_tree_path_free(path2);
     }
     if (success)
          save_effect_order(eb);
}

static void move_top_bottom_main(EffectBrowser *eb, gboolean top)
{
     GtkTreeModel *model;
     GtkTreeIter iter;
     if (gtk_tree_selection_get_selected (eb->list_sel, &model, &iter)) {
          if (top)
               gtk_list_store_move_after (eb->liststore, &iter, NULL);
          else
               gtk_list_store_move_before (eb->liststore, &iter, NULL);
     }
     save_effect_order(eb);
}

static void list_item_moveup(GtkWidget *widget, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     move_up_down_main(eb,TRUE);
}

static void list_item_movedown(GtkWidget *widget, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     move_up_down_main(eb,FALSE);
}

static void list_item_movetotop(GtkWidget *widget, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     move_top_bottom_main(eb,TRUE);
}

static void list_item_movetobottom(GtkWidget *widget, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     move_top_bottom_main(eb,FALSE);
}

static void list_item_sort_main(EffectBrowser *eb, GCompareFunc compfunc)
{
     /* Not the quickest way, but preserves original order if compfunc 
	returns >0 when objects are equal */
     GList *l,*m=NULL;
     gint i;
     struct effect *e;
     gchar *c,*d;
     GtkTreeModel *model;
     GtkTreeIter iter;
     gboolean valid;

     model = GTK_TREE_MODEL(eb->liststore);
     valid = gtk_tree_model_get_iter_first(model,&iter);
     while(valid) {
          gtk_tree_model_get (model, &iter, 1, &e, -1);
	  g_assert(e != NULL);
	  m = g_list_insert_sorted(m,e,compfunc);
          valid = gtk_tree_model_iter_next (model, &iter);
     }

     for (l=m,i=0; l!=NULL; l=l->next,i++) {
	  e = (struct effect *)l->data;
	  c = g_strdup_printf("effectBrowserOrder%d",i);
	  d = g_strdup_printf("%c%s",e->source_tag,e->name);
	  inifile_set(c,d);
	  g_free(d);
	  g_free(c);
     }
     c = g_strdup_printf("effectBrowserOrder%d",i);
     inifile_set(c,NULL);
     g_free(c);
     g_list_free(m);
     list_widget_rebuild(NULL,NULL,eb);
}

gint title_sort_func(gconstpointer a, gconstpointer b)
{
     struct effect const *ae = a, *be = b;
     int i;
     i = strcmp(ae->title,be->title);
     if (i==0) return 1; else return i;
}

gint auth_sort_func(gconstpointer a, gconstpointer b)
{
     struct effect const *ae = a, *be = b;
     int i;
     i = strcmp(ae->author,be->author);
     if (i==0) return 1; else return i;
}

gint type_sort_func(gconstpointer a, gconstpointer b)
{
     struct effect const *ae = a, *be = b;
     int i;
     i = ae->source_tag - be->source_tag;
     if (i==0) return 1; else return i;
}

gint loc_sort_func(gconstpointer a, gconstpointer b)
{
     struct effect const *ae = a, *be = b;
     int i;
     i = strcmp(ae->location,be->location);
     if (i==0) return 1; else return i;
}


static void list_item_sortbytitle(GtkWidget *widget, gpointer user_data)
{     
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     list_item_sort_main(eb, title_sort_func);
}

static void list_item_sortbytype(GtkWidget *widget, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     list_item_sort_main(eb, type_sort_func);
}

static void list_item_sortbyloc(GtkWidget *widget, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     list_item_sort_main(eb, loc_sort_func);
}

static void list_item_sortbyauth(GtkWidget *widget, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     list_item_sort_main(eb, auth_sort_func);
}

static void list_item_unsort(GtkWidget *widget, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     inifile_set("effectBrowserOrder0",NULL);
     list_widget_rebuild(NULL,NULL,eb);
}

static void list_item_rebuild(GtkWidget *widget, gpointer user_data)
{
     effect_register_rebuild();
}

static gboolean list_item_button_press(GtkWidget *widget, GdkEventButton *event,
				       gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     GtkWidget *menu, *item;

     if (event->button == 3) {
          menu = gtk_menu_new();
          item = gtk_menu_item_new_with_label(_("Move Up"));
          g_signal_connect(item, "activate",G_CALLBACK(list_item_moveup),eb);
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_menu_item_new_with_label(_("Move Down"));
          g_signal_connect(item, "activate",G_CALLBACK(list_item_movedown),eb);
           gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_menu_item_new_with_label(_("Move to Top"));
          g_signal_connect(item, "activate",G_CALLBACK(list_item_movetotop),eb);
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_menu_item_new_with_label(_("Move to Bottom"));
          g_signal_connect(item, "activate",G_CALLBACK(list_item_movetobottom),eb);
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_separator_menu_item_new ();
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_menu_item_new_with_label(_("Sort by Name"));
          g_signal_connect(item, "activate",G_CALLBACK(list_item_sortbytitle),eb);
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_menu_item_new_with_label(_("Sort by Type"));
          g_signal_connect(item, "activate",G_CALLBACK(list_item_sortbytype),eb);
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_menu_item_new_with_label(_("Sort by Location"));
          g_signal_connect(item, "activate",G_CALLBACK(list_item_sortbyloc),eb);
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_menu_item_new_with_label(_("Sort by Author"));
          g_signal_connect(item, "activate",G_CALLBACK(list_item_sortbyauth),eb);
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_separator_menu_item_new ();
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_menu_item_new_with_label(_("Restore Order"));
          g_signal_connect(item, "activate",G_CALLBACK(list_item_unsort),eb);
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          item = gtk_menu_item_new_with_label(_("Rebuild Effect List"));
          g_signal_connect(item, "activate",G_CALLBACK(list_item_rebuild),eb);
          gtk_menu_shell_append(GTK_MENU_SHELL(menu), item);
          gtk_menu_attach_to_widget (GTK_MENU (menu), widget, NULL);
          gtk_widget_show_all(menu);
          gtk_menu_popup(GTK_MENU(menu),NULL,NULL,NULL,NULL,event->button,
			 event->time);
          return TRUE;
     }
     return FALSE;
}

static void add_list_item_main(struct effect *e, GtkListStore *l, EffectBrowser *eb)
{
     gchar *c,*d;
     GtkTreeIter iter;
     c = g_strdup_printf("[%c] %s",e->source_tag,e->title);

     /* Translate here for keeping compatibility with old translations */
     /* New translations should translate the title without the prefix */
     if (e->source_tag == 'B' || e->source_tag == 'S') d = _(c); else d = c;

     gtk_list_store_append(l,&iter);
     gtk_list_store_set(l,&iter,0,d,1,e,-1);
     g_free(c);
}

static void add_list_item(gpointer item, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     struct effect *e = (struct effect *)item;
     add_list_item_main(e,eb->liststore,eb);
}

static void top_click(GtkButton *button, gpointer user_data)
{     
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     move_top_bottom_main(eb,TRUE);
}

static void bottom_click(GtkButton *button, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     move_top_bottom_main(eb,FALSE);
}

static void up_click(GtkButton *button, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     move_up_down_main(eb,TRUE);
}

static void down_click(GtkButton *button, gpointer user_data)
{
     EffectBrowser *eb = EFFECT_BROWSER(user_data);
     move_up_down_main(eb,FALSE);
}

static void add_list_widget_items(GtkListStore *list, EffectBrowser *eb)
{
     gint i;
     gchar *c,*d;
     GList *l;
     struct effect *e;
     if (inifile_get("effectBrowserOrder0",NULL) == NULL) {
	  list_object_foreach(effect_list,add_list_item,eb);
     } else {
	  for (l=effect_list->list; l!=NULL; l=l->next) {
	       e = (struct effect *)l->data;
	       e->process_tag = FALSE;
	  }
	  for (i=0; ; i++) {
	       c = g_strdup_printf("effectBrowserOrder%d",i);
	       d = inifile_get(c,NULL);
	       g_free(c);
	       if (d == NULL) break;
	       for (l=effect_list->list; l!=NULL; l=l->next) {
		    e = (struct effect *)l->data;
		    if (e->process_tag) continue;
		    if (e->source_tag != d[0] || strcmp(e->name,d+1)) continue;
		    add_list_item_main(e,list,eb);
		    e->process_tag = TRUE;
		    break;
	       }
	  }
	  for (l=effect_list->list; l!=NULL; l=l->next) {
	       e = (struct effect *)l->data;
	       if (!e->process_tag)
		    add_list_item_main(e,list,eb);
	  }
     }
}

static void list_widget_rebuild(gpointer dummy, gpointer dummy2, 
				EffectBrowser *eb)
{
     gtk_list_store_clear(eb->liststore);
     add_list_widget_items(eb->liststore, eb);
}

static void effect_browser_init(EffectBrowser *eb)
{
     GtkWidget *b,*b1,*b11,*b11w,*b12,*b121,*b122,*b123,*b124,*b2,*b21;
     GtkWidget *b21w,*b22,*b23,*b24,*b25,*b251,*b252;
     GtkAccelGroup* ag;
     gchar *c,*d;
     gint x;
     GtkTreeViewColumn *col;
     GtkCellRenderer *renderer;

     ag = gtk_accel_group_new();

     memset(eb->dialogs,0,sizeof(eb->dialogs));
     memset(eb->dialog_effects,0,sizeof(eb->dialog_effects));
     eb->current_dialog = -1;
     
     b11w = gtk_tree_view_new();
     gtk_tree_view_set_headers_visible(GTK_TREE_VIEW(b11w),FALSE);
     renderer = gtk_cell_renderer_text_new();
     col = gtk_tree_view_column_new_with_attributes(NULL,renderer,"text",0,NULL);
     gtk_tree_view_append_column(GTK_TREE_VIEW(b11w),col);
     eb->list_sel = gtk_tree_view_get_selection(GTK_TREE_VIEW(b11w));
     gtk_tree_selection_set_mode(eb->list_sel,GTK_SELECTION_SINGLE);
     eb->liststore = gtk_list_store_new(2,G_TYPE_STRING,G_TYPE_POINTER);
     gtk_tree_view_set_model(GTK_TREE_VIEW(b11w),GTK_TREE_MODEL(eb->liststore));
     g_signal_connect(b11w,"button_press_event",
                      G_CALLBACK(list_item_button_press),eb);

     effect_register_update_list();
     add_list_widget_items(eb->liststore,eb);

     g_signal_connect(G_OBJECT(effect_list),"item-notify",
			G_CALLBACK(list_widget_rebuild),eb);

     b11 = gtk_scrolled_window_new(NULL,NULL);
     gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(b11),
				    GTK_POLICY_AUTOMATIC,GTK_POLICY_AUTOMATIC);
     gtk_container_add (GTK_CONTAINER (b11), b11w);
     gtk_widget_set_size_request(GTK_WIDGET(b11),150,150);
     eb->effect_list_container = GTK_CONTAINER(b11);

#ifdef GTK_STOCK_GOTO_TOP
     b121 = gtk_button_new_from_stock(GTK_STOCK_GOTO_TOP);
#else
     b121 = gtk_button_new_with_label(_("Top"));
#endif
     g_signal_connect(G_OBJECT(b121),"clicked",
			G_CALLBACK(top_click),eb);
#ifdef GTK_STOCK_GO_UP
     b122 = gtk_button_new_from_stock(GTK_STOCK_GO_UP);
#else
     b122 = gtk_button_new_with_label(_("Up"));
#endif
     g_signal_connect(G_OBJECT(b122),"clicked",
			G_CALLBACK(up_click),eb);
#ifdef GTK_STOCK_GO_DOWN
     b123 = gtk_button_new_from_stock(GTK_STOCK_GO_DOWN);
#else
     b123 = gtk_button_new_with_label(_("Down"));
#endif
     g_signal_connect(G_OBJECT(b123),"clicked",
			G_CALLBACK(down_click),eb);
#ifdef GTK_STOCK_GOTO_BOTTOM
     b124 = gtk_button_new_from_stock(GTK_STOCK_GOTO_BOTTOM);
#else
     b124 = gtk_button_new_with_label(_("Bottom"));
#endif
     g_signal_connect(G_OBJECT(b124),"clicked",
			G_CALLBACK(bottom_click),eb);

     b12 = gtk_hbox_new(FALSE,5);
     gtk_box_pack_start(GTK_BOX(b12),b121,FALSE,FALSE,0);
     gtk_box_pack_start(GTK_BOX(b12),b122,FALSE,FALSE,0);
     gtk_box_pack_start(GTK_BOX(b12),b123,FALSE,FALSE,0);
     gtk_box_pack_start(GTK_BOX(b12),b124,FALSE,FALSE,0);

     b1 = gtk_vbox_new(FALSE,5);
     gtk_box_pack_start(GTK_BOX(b1),b11,TRUE,TRUE,0);
     gtk_box_pack_start(GTK_BOX(b1),b12,FALSE,FALSE,0);

     b21w = gtk_alignment_new(0.5,0.5,1.0,1.0);
     eb->dialog_container = GTK_CONTAINER(b21w);
     
     b21 = gtk_scrolled_window_new(NULL,NULL);
     gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(b21),
				    GTK_POLICY_AUTOMATIC,GTK_POLICY_AUTOMATIC);
     gtk_scrolled_window_add_with_viewport(GTK_SCROLLED_WINDOW(b21),b21w);
  
     b22 = gtk_hseparator_new();

     b23 = gtk_hbox_new(FALSE,3);
     eb->mw_list_box = GTK_BOX(b23);

     b24 = gtk_check_button_new_with_label(_("Close dialog after applying "
					     "effect"));
     eb->close_after = GTK_TOGGLE_BUTTON(b24);

#ifdef GTK_STOCK_APPLY
     b251 = gtk_button_new_from_stock(GTK_STOCK_APPLY);
#else
     b251 = gtk_button_new_with_label(_("Apply"));
#endif
     g_signal_connect(G_OBJECT(b251),"clicked",G_CALLBACK(apply_click),
			eb);

#ifdef GTK_STOCK_CLOSE
     b252 = gtk_button_new_from_stock(GTK_STOCK_CLOSE);
#else
     b252 = gtk_button_new_with_label(_("Close"));
#endif
     gtk_widget_add_accelerator (b252, "clicked", ag, GDK_KEY(Escape), 0, (GtkAccelFlags) 0);
     g_signal_connect_swapped(G_OBJECT(b252),"clicked",
			       G_CALLBACK(effect_browser_close), eb);

     b25 = gtk_hbutton_box_new(); 
     gtk_button_box_set_layout(GTK_BUTTON_BOX(b25),GTK_BUTTONBOX_END);
     gtk_box_pack_start(GTK_BOX(b25),b251,FALSE,TRUE,3);
     gtk_box_pack_start(GTK_BOX(b25),b252,FALSE,TRUE,3);

     b2 = gtk_vbox_new(FALSE,5);     
     gtk_box_pack_start(GTK_BOX(b2),b21,TRUE,TRUE,0);
     gtk_box_pack_end(GTK_BOX(b2),b25,FALSE,FALSE,0);
     gtk_box_pack_end(GTK_BOX(b2),b24,FALSE,FALSE,0);
     gtk_box_pack_end(GTK_BOX(b2),b23,FALSE,TRUE,0);
     gtk_box_pack_end(GTK_BOX(b2),b22,FALSE,TRUE,0);

     b = gtk_hpaned_new();
     gtk_paned_pack1(GTK_PANED(b),b1,FALSE,TRUE);
     gtk_paned_pack2(GTK_PANED(b),b2,TRUE,TRUE);

     gtk_window_set_title(GTK_WINDOW(eb),_("Effects"));
     gtk_window_add_accel_group(GTK_WINDOW (eb), ag);
     gtk_window_set_resizable(GTK_WINDOW(eb),TRUE);

     if (!geometry_stack_inited) {
	  if (inifile_get_gboolean("useGeometry",FALSE))
	       geometry_stack = geometry_stack_from_inifile("effectGeometry");
	  geometry_stack_inited = TRUE;
     }
     if (!geometry_stack_pop(&geometry_stack,&c,GTK_WINDOW(eb))) {
	 gtk_window_set_position (GTK_WINDOW (eb), GTK_WIN_POS_CENTER);
	 gtk_window_set_default_size(GTK_WINDOW(eb),600,300);	 
     } else {
	  if (c != NULL) {
	       x = strtoul(c,&d,10);
	       if (*d == 0 && *c != 0)
		    gtk_paned_set_position(GTK_PANED(b),x);	       
	       g_free(c);
	  }
     }
     gtk_container_set_border_width(GTK_CONTAINER(eb),5);
     gtk_container_add(GTK_CONTAINER(eb),b);
     gtk_widget_set_can_default (b251, TRUE);
     gtk_widget_grab_default(GTK_WIDGET(b251));
     gtk_widget_show_all(b);
}

GtkWidget *effect_browser_new(Document *doc)
{
     return effect_browser_new_with_effect(doc,"volume",'B',FALSE);
}

GtkWidget *effect_browser_new_with_effect(Document *doc, gchar *effect, 
					  gchar source_tag, 
					  gboolean close_after)
{
     GtkWidget *w;
     EffectBrowser *eb = g_object_new(EFFECT_BROWSER_TYPE, NULL);
     g_signal_connect(eb->list_sel,"changed",
		      G_CALLBACK(effect_browser_selection_changed),eb);

     w = document_list_new(doc);
     gtk_box_pack_end(GTK_BOX(eb->mw_list_box),w,TRUE,TRUE,0);
     gtk_widget_show(w);
     eb->dl = DOCUMENT_LIST(w);
     w = gtk_label_new(_("Apply to: "));
     gtk_box_pack_end(GTK_BOX(eb->mw_list_box),w,FALSE,FALSE,0);
     gtk_widget_show(w);     

     if (effect == NULL) {
	  effect = inifile_get("lastEffect","Bvolume");
	  source_tag = effect[0];
	  effect++;
     }
     effect_browser_set_effect(eb,effect,source_tag);
     if (eb->current_dialog < 0) effect_browser_set_effect(eb,"volume",'B');
     g_assert(eb->current_dialog >= 0);
     gtk_toggle_button_set_active(eb->close_after,close_after);
     return GTK_WIDGET(eb);
}

void effect_browser_set_effect(EffectBrowser *eb, gchar *effect, 
			       gchar source_tag)
{
     struct effect *e;
     GList *l;
     gpointer p;
     gboolean valid;
     GtkTreeModel *model;
     GtkTreeIter iter;
     model = GTK_TREE_MODEL(eb->liststore);
     for (l=effect_list->list; l!=NULL; l=l->next) {
	  e = (struct effect *)l->data;
	  if (e->source_tag == source_tag && !strcmp(e->name, effect)) {
	       /* Find the list item which points to this effect */
               valid = gtk_tree_model_get_iter_first(model,&iter);
	       while(valid) {
                    gtk_tree_model_get (model, &iter, 1, &p, -1);
		    g_assert(p != NULL);
		    if (p == e) {
			 gtk_tree_selection_select_iter(eb->list_sel,&iter);
			 return;
		    }
                    valid = gtk_tree_model_iter_next (model, &iter);
	       }
	       /* Effect exists but not in list, shouldn't happen */
	       g_assert_not_reached();
	  }	  
     }
     /* Effect doesn't exist - do nothing */
}

void effect_browser_shutdown(void)
{     
     if (inifile_get_gboolean("useGeometry",FALSE))
	  geometry_stack_save_to_inifile("effectGeometry",geometry_stack);  
}

