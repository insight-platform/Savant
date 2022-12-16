#include "dsrbbox_meta.h"
#include <iostream>
#include "user_meta_type.h"

NvRBboxCoords* get_rbbox(NvDsObjectMeta *object_meta){
    NvDsUserMetaList *user_meta_list, *tmp_user_meta_list;
    user_meta_list = object_meta -> obj_user_meta_list;
    while ((user_meta_list != nullptr) && (((NvDsUserMeta* ) user_meta_list->data)->base_meta.meta_type != NVDS_USER_META_ROTATED_BBOX))
    {
        user_meta_list = user_meta_list->next;
    }
    if (user_meta_list)
    {
        tmp_user_meta_list = user_meta_list->next;
        while ((tmp_user_meta_list != nullptr) && (((NvDsUserMeta* ) tmp_user_meta_list->data)->base_meta.meta_type != NVDS_USER_META_ROTATED_BBOX))
        {
            tmp_user_meta_list = tmp_user_meta_list->next;
        }
        if (tmp_user_meta_list != nullptr)
        {
            GST_ERROR("User meta for the object id `%lu` contains more then one rotated bounding box", object_meta -> object_id);
        }
        return (NvRBboxCoords*) ((NvDsUserMeta* ) user_meta_list -> data) ->user_meta_data;    
    }
    return nullptr;
}

/* copy function set by user. "data" holds a pointer to NvDsUserMeta*/
static gpointer copy_user_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  gchar *src_user_metadata = (gchar*)user_meta->user_meta_data;
  gchar *dst_user_metadata = (gchar*)g_malloc0(sizeof(NvRBboxCoords));
  memcpy(dst_user_metadata, src_user_metadata, sizeof(NvRBboxCoords));
  return (gpointer)dst_user_metadata;
}

/* release function set by user. "data" holds a pointer to NvDsUserMeta*/
static void release_user_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  if(user_meta->user_meta_data) {
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
  }
}

NvRBboxCoords* acquire_rbbox(){
    return g_new0(NvRBboxCoords, 1);
}

void add_rbbox_to_object_meta(NvDsBatchMeta *batch_meta, NvDsObjectMeta* obj_meta, NvRBboxCoords* rbbox){
    
    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    /* Set NvDsUserMeta below */
    gchar *dst_user_metadata = (gchar*)g_malloc0(sizeof(NvRBboxCoords));
    memcpy(dst_user_metadata, rbbox, sizeof(NvRBboxCoords));

    rbbox = (NvRBboxCoords *) dst_user_metadata;
    user_meta->user_meta_data = (void *) dst_user_metadata;
    user_meta->base_meta.meta_type = NVDS_USER_META_ROTATED_BBOX;
    user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)copy_user_meta;
    user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_user_meta;
    nvds_add_user_meta_to_obj(obj_meta, user_meta);
}

RBBoxIterator::RBBoxIterator(const RBBoxIterator& it)
{
    user_meta_item = it.user_meta_item;
}

RBBoxIterator::RBBoxIterator(NvDsUserMetaList *meta_list)
{
    NvDsUserMeta* user_meta;
    user_meta_item = meta_list;
    while ((user_meta_item != nullptr) && (((NvDsUserMeta* ) user_meta_item->data)->base_meta.meta_type != NVDS_USER_META_ROTATED_BBOX))
    {
      user_meta_item = user_meta_item->next;
    } 
}

bool RBBoxIterator::operator!=(RBBoxIterator const& other) const
{
    return  user_meta_item != other.user_meta_item;
}

bool RBBoxIterator::operator==(RBBoxIterator const& other) const
{
    return user_meta_item == other.user_meta_item;
}

NvRBboxCoords* RBBoxIterator::operator*() const
{
    return (NvRBboxCoords*) ((NvDsUserMeta *) user_meta_item->data)->user_meta_data;
}

RBBoxIterator& RBBoxIterator::operator++()
{
    user_meta_item = user_meta_item->next;
    while ((user_meta_item != nullptr) && (((NvDsUserMeta* ) user_meta_item->data)->base_meta.meta_type != NVDS_USER_META_ROTATED_BBOX))
    {
      user_meta_item = user_meta_item->next;
    }
    
    return *this;
}