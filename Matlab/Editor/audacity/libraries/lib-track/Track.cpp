/**********************************************************************

  Audacity: A Digital Audio Editor

  Track.cpp

  Dominic Mazzoni

*******************************************************************//**

\class Track
\brief Fundamental data object of Audacity, displayed in the TrackPanel.
Classes derived form it include the WaveTrack, NoteTrack, LabelTrack
and TimeTrack.

\class AudioTrack
\brief A Track that can load/save audio data to/from XML.

\class PlayableTrack
\brief An AudioTrack that can be played and stopped.

*//*******************************************************************/

#include "Track.h"

#include <algorithm>
#include <numeric>

#include <float.h>
#include <wx/file.h>
#include <wx/textfile.h>
#include <wx/log.h>

#include "BasicUI.h"
#include "Project.h"

#include "InconsistencyException.h"

#ifdef _MSC_VER
//Disable truncation warnings
#pragma warning( disable : 4786 )
#endif

Track::Track()
:  vrulerSize(36,0)
{
   mSelected  = false;

   mIndex = 0;

   mOffset = 0.0;

   mChannel = MonoChannel;
}

Track::Track(const Track &orig, ProtectedCreationArg&&)
: vrulerSize( orig.vrulerSize )
{
   mIndex = 0;
   mOffset = orig.mOffset;
}

// Copy all the track properties except the actual contents
void Track::Init(const Track &orig)
{
   mId = orig.mId;

   mName = orig.mName;

   mSelected = orig.mSelected;

   // Deep copy of any group data
   mpGroupData = orig.mpGroupData ?
      std::make_unique<ChannelGroupData>(*orig.mpGroupData) : nullptr;

   mChannel = orig.mChannel;
}

void Track::SetName( const wxString &n )
{
   if ( mName != n ) {
      mName = n;
      Notify();
   }
}

void Track::SetSelected(bool s)
{
   if (mSelected != s) {
      mSelected = s;
      auto pList = mList.lock();
      if (pList)
         pList->SelectionEvent( SharedPointer() );
   }
}

void Track::EnsureVisible( bool modifyState )
{
   auto pList = mList.lock();
   if (pList)
      pList->EnsureVisibleEvent( SharedPointer(), modifyState );
}

void Track::Merge(const Track &orig)
{
   mSelected = orig.mSelected;
}

Track::Holder Track::Duplicate() const
{
   // invoke "virtual constructor" to copy track object proper:
   auto result = Clone();

   AttachedTrackObjects::ForEach([&](auto &attachment){
      // Copy view state that might be important to undo/redo
      attachment.CopyTo( *result );
   });

   return result;
}

Track::~Track()
{
}


TrackNodePointer Track::GetNode() const
{
   wxASSERT(mList.lock() == NULL || this == mNode.first->get());
   return mNode;
}

void Track::SetOwner
(const std::weak_ptr<TrackList> &list, TrackNodePointer node)
{
   // BUG: When using this function to clear an owner, we may need to clear
   // focused track too.  Otherwise focus could remain on an invisible (or deleted) track.
   mList = list;
   mNode = node;
}

int Track::GetIndex() const
{
   return mIndex;
}

void Track::SetIndex(int index)
{
   mIndex = index;
}

void Track::SetLinkType(LinkType linkType, bool completeList)
{
   auto pList = mList.lock();
   if (pList && !pList->mPendingUpdates.empty()) {
      auto orig = pList->FindById( GetId() );
      if (orig && orig != this) {
         orig->SetLinkType(linkType);
         return;
      }
   }

   DoSetLinkType(linkType, completeList);

   if (pList) {
      pList->RecalcPositions(mNode);
      pList->ResizingEvent(mNode);
   }
}

Track::ChannelGroupData &Track::MakeGroupData()
{
   if (!mpGroupData)
      // Make on demand
      mpGroupData = std::make_unique<ChannelGroupData>();
   return *mpGroupData;
}

Track::ChannelGroupData &Track::GetGroupData()
{
   auto pTrack = this;
   if (auto pList = GetOwner())
      if (auto pLeader = *pList->FindLeader(pTrack))
         pTrack = pLeader;
   // May make on demand
   return pTrack->MakeGroupData();
}

const Track::ChannelGroupData &Track::GetGroupData() const
{
   // May make group data on demand, but consider that logically const
   return const_cast<Track *>(this)->GetGroupData();
}

void Track::DoSetLinkType(LinkType linkType, bool completeList)
{
   auto oldType = GetLinkType();
   if (linkType == oldType)
      // No change
      return;

   if (oldType == LinkType::None) {
      // Becoming linked
   
      // First ensure there is no partner
      if (auto partner = GetLinkedTrack())
         partner->mpGroupData.reset();
      assert(!GetLinkedTrack());
   
      // Change the link type
      MakeGroupData().mLinkType = linkType;

      // If this acquired a partner, it loses any old group data
      if (auto partner = GetLinkedTrack())
         partner->mpGroupData.reset();
   }
   else if (linkType == LinkType::None) {
      // Becoming unlinked
      assert(mpGroupData);
      if (HasLinkedTrack()) {
         if (auto partner = GetLinkedTrack()) {
            // Make independent copy of group data in the partner, which should
            // have had none
            assert(!partner->mpGroupData);
            partner->mpGroupData =
               std::make_unique<ChannelGroupData>(*mpGroupData);
            partner->mpGroupData->mLinkType = LinkType::None;
         }
      }
      mpGroupData->mLinkType = LinkType::None;
   }
   else {
      // Remaining linked, changing the type
      assert(mpGroupData);
      MakeGroupData().mLinkType = linkType;
   }

   // Assertion checks only in a debug build, does not have side effects!
   assert(LinkConsistencyCheck(completeList));
}

void Track::SetChannel(ChannelType c) noexcept
{
    mChannel = c;
}

Track *Track::GetLinkedTrack() const
{
   auto pList = mList.lock();
   if (!pList)
      return nullptr;

   if (!pList->isNull(mNode)) {
      if (HasLinkedTrack()) {
         auto next = pList->getNext( mNode );
         if ( !pList->isNull( next ) )
            return next.first->get();
      }

      if (mNode.first != mNode.second->begin()) {
         auto prev = pList->getPrev( mNode );
         if ( !pList->isNull( prev ) ) {
            auto track = prev.first->get();
            if (track && track->HasLinkedTrack())
               return track;
         }
      }
   }

   return nullptr;
}

bool Track::HasLinkedTrack() const noexcept
{
    return mpGroupData && mpGroupData->mLinkType != LinkType::None;
}

void Track::Notify( int code )
{
   auto pList = mList.lock();
   if (pList)
      pList->DataEvent( SharedPointer(), code );
}

void Track::SyncLockAdjust(double oldT1, double newT1)
{
   if (newT1 > oldT1) {
      // Insert space within the track

      if (oldT1 > GetEndTime())
         return;

      auto tmp = Cut(oldT1, GetEndTime());

      Paste(newT1, tmp.get());
   }
   else if (newT1 < oldT1) {
      // Remove from the track
      Clear(newT1, oldT1);
   }
}

AudioTrack::AudioTrack() : Track{}
{
}

AudioTrack::AudioTrack(const Track &orig, ProtectedCreationArg &&a)
   : Track{ orig, std::move(a) }
{
}

PlayableTrack::PlayableTrack() : AudioTrack{}
{
}

PlayableTrack::PlayableTrack(
   const PlayableTrack &orig, ProtectedCreationArg &&a
)  : AudioTrack{ orig, std::move(a) }
{
}

void PlayableTrack::Init( const PlayableTrack &orig )
{
   DoSetMute(orig.DoGetMute());
   DoSetSolo(orig.DoGetSolo());
   AudioTrack::Init( orig );
}

void PlayableTrack::Merge( const Track &orig )
{
   auto pOrig = dynamic_cast<const PlayableTrack *>(&orig);
   wxASSERT( pOrig );
   DoSetMute(pOrig->DoGetMute());
   DoSetSolo(pOrig->DoGetSolo());
   AudioTrack::Merge( *pOrig );
}

void PlayableTrack::SetMute( bool m )
{
   if ( DoGetMute() != m ) {
      DoSetMute(m);
      Notify();
   }
}

void PlayableTrack::SetSolo( bool s  )
{
   if ( DoGetSolo() != s ) {
      DoSetSolo(s);
      Notify();
   }
}

bool PlayableTrack::DoGetMute() const
{
   return mMute.load(std::memory_order_relaxed);
}

void PlayableTrack::DoSetMute(bool value)
{
   mMute.store(value, std::memory_order_relaxed);
}

bool PlayableTrack::DoGetSolo() const
{
   return mSolo.load(std::memory_order_relaxed);
}

void PlayableTrack::DoSetSolo(bool value)
{
   mSolo.store(value, std::memory_order_relaxed);
}

// Serialize, not with tags of its own, but as attributes within a tag.
void PlayableTrack::WriteXMLAttributes(XMLWriter &xmlFile) const
{
   xmlFile.WriteAttr(wxT("mute"), DoGetMute());
   xmlFile.WriteAttr(wxT("solo"), DoGetSolo());
   AudioTrack::WriteXMLAttributes(xmlFile);
}

// Return true iff the attribute is recognized.
bool PlayableTrack::HandleXMLAttribute(const std::string_view &attr, const XMLAttributeValueView &value)
{
   long nValue;

   if (attr == "mute" && value.TryGet(nValue)) {
      DoSetMute(nValue != 0);
      return true;
   }
   else if (attr == "solo" && value.TryGet(nValue)) {
      DoSetSolo(nValue != 0);
      return true;
   }

   return AudioTrack::HandleXMLAttribute(attr, value);
}

bool Track::Any() const
   { return true; }

bool Track::IsSelected() const
   { return GetSelected(); }

bool Track::IsLeader() const
{
    return !GetLinkedTrack() || HasLinkedTrack();
}

bool Track::IsSelectedLeader() const
   { return IsSelected() && IsLeader(); }

void Track::FinishCopy
(const Track *n, Track *dest)
{
   if (dest) {
      dest->SetChannel(n->GetChannel());
      dest->mpGroupData = n->mpGroupData ?
         std::make_unique<ChannelGroupData>(*n->mpGroupData) : nullptr;
      dest->SetName(n->GetName());
   }
}

bool Track::LinkConsistencyFix(bool doFix, bool completeList)
{
   // Sanity checks for linked tracks; unsetting the linked property
   // doesn't fix the problem, but it likely leaves us with orphaned
   // sample blocks instead of much worse problems.
   bool err = false;
   if (completeList && HasLinkedTrack()) {
      if (auto link = GetLinkedTrack()) {
         // A linked track's partner should never itself be linked
         if (link->HasLinkedTrack()) {
            err = true;
            if (doFix) {
               wxLogWarning(
                  L"Left track %s had linked right track %s with extra right "
                   "track link.\n   Removing extra link from right track.",
                  GetName(), link->GetName());
               link->SetLinkType(LinkType::None);
            }
         }

         // Channels should be left and right
         if ( !(  (GetChannel() == Track::LeftChannel &&
                     link->GetChannel() == Track::RightChannel) ||
                  (GetChannel() == Track::RightChannel &&
                     link->GetChannel() == Track::LeftChannel) ) ) {
            err = true;
            if (doFix) {
               wxLogWarning(
                  L"Track %s and %s had left/right track links out of order. "
                   "Setting tracks to not be linked.",
                  GetName(), link->GetName());
               SetLinkType(LinkType::None);
            }
         }
      }
      else {
         err = true;
         if (doFix) {
            wxLogWarning(
               L"Track %s had link to NULL track. Setting it to not be linked.",
               GetName());
            SetLinkType(LinkType::None);
         }
      }
   }
   return ! err;
}

// TrackList
//
// The TrackList sends events whenever certain updates occur to the list it
// is managing.  Any other classes that may be interested in get these updates
// should use TrackList::Subscribe().
//

// same value as in the default constructed TrackId:
long TrackList::sCounter = -1;

static const AudacityProject::AttachedObjects::RegisteredFactory key{
   [](AudacityProject &project) { return TrackList::Create( &project ); }
};

TrackList &TrackList::Get( AudacityProject &project )
{
   return project.AttachedObjects::Get< TrackList >( key );
}

const TrackList &TrackList::Get( const AudacityProject &project )
{
   return Get( const_cast< AudacityProject & >( project ) );
}

TrackList::TrackList( AudacityProject *pOwner )
   : mOwner{ pOwner }
{
}

// Factory function
std::shared_ptr<TrackList> TrackList::Create( AudacityProject *pOwner )
{
   return std::make_shared<TrackList>( pOwner );
}

#if 0
TrackList &TrackList::operator= (TrackList &&that)
{
   if (this != &that) {
      this->Clear();
      Swap(that);
   }
   return *this;
}
#endif

void TrackList::Swap(TrackList &that)
{
   auto SwapLOTs = [](
      ListOfTracks &a, const std::weak_ptr< TrackList > &aSelf,
      ListOfTracks &b, const std::weak_ptr< TrackList > &bSelf )
   {
      a.swap(b);
      for (auto it = a.begin(), last = a.end(); it != last; ++it)
         (*it)->SetOwner(aSelf, {it, &a});
      for (auto it = b.begin(), last = b.end(); it != last; ++it)
         (*it)->SetOwner(bSelf, {it, &b});
   };

   const auto self = shared_from_this();
   const auto otherSelf = that.shared_from_this();
   SwapLOTs( *this, self, that, otherSelf );
   SwapLOTs( this->mPendingUpdates, self, that.mPendingUpdates, otherSelf );
   mUpdaters.swap(that.mUpdaters);
}

TrackList::~TrackList()
{
   Clear(false);
}

wxString TrackList::MakeUniqueTrackName(const wxString& baseTrackName) const
{
   int n = 1;
   while(true)
   {
      auto name = wxString::Format("%s %d", baseTrackName, n++);

      bool found {false};
      for(const auto track : Any())
      {
         if(track->GetName() == name)
         {
            found = true;
            break;
         }
      }
      if(!found)
         return name;
   }
}

void TrackList::RecalcPositions(TrackNodePointer node)
{
   if ( isNull( node ) )
      return;

   Track *t;
   int i = 0;

   auto prev = getPrev( node );
   if ( !isNull( prev ) ) {
      t = prev.first->get();
      i = t->GetIndex() + 1;
   }

   const auto theEnd = end();
   for (auto n = Find( node.first->get() ); n != theEnd; ++n) {
      t = *n;
      t->SetIndex(i++);
   }

   UpdatePendingTracks();
}

void TrackList::QueueEvent(TrackListEvent event)
{
   BasicUI::CallAfter( [wThis = weak_from_this(), event = std::move(event)]{
      if (auto pThis = wThis.lock())
         pThis->Publish(event);
   } );
}

void TrackList::SelectionEvent( const std::shared_ptr<Track> &pTrack )
{
   QueueEvent({ TrackListEvent::SELECTION_CHANGE, pTrack });
}

void TrackList::DataEvent( const std::shared_ptr<Track> &pTrack, int code )
{
   QueueEvent({
      TrackListEvent::TRACK_DATA_CHANGE, pTrack, code });
}

void TrackList::EnsureVisibleEvent(
   const std::shared_ptr<Track> &pTrack, bool modifyState )
{
   QueueEvent({ TrackListEvent::TRACK_REQUEST_VISIBLE,
      pTrack, static_cast<int>(modifyState) });
}

void TrackList::PermutationEvent(TrackNodePointer node)
{
   QueueEvent({ TrackListEvent::PERMUTED, *node.first });
}

void TrackList::DeletionEvent(std::weak_ptr<Track> node, bool duringReplace)
{
   QueueEvent(
      { TrackListEvent::DELETION, std::move(node), duringReplace ? 1 : 0 });
}

void TrackList::AdditionEvent(TrackNodePointer node)
{
   QueueEvent({ TrackListEvent::ADDITION, *node.first });
}

void TrackList::ResizingEvent(TrackNodePointer node)
{
   QueueEvent({ TrackListEvent::RESIZING, *node.first });
}

auto TrackList::EmptyRange() const
   -> TrackIterRange< Track >
{
   auto it = const_cast<TrackList*>(this)->getEnd();
   return {
      { it, it, it, &Track::Any },
      { it, it, it, &Track::Any }
   };
}

auto TrackList::FindLeader( Track *pTrack )
   -> TrackIter< Track >
{
   auto iter = Find(pTrack);
   while( *iter && ! ( *iter )->IsLeader() )
      --iter;
   return iter.Filter( &Track::IsLeader );
}

bool TrackList::SwapChannels(Track &track)
{
   if (!track.HasLinkedTrack())
      return false;
   auto pOwner = track.GetOwner();
   if (!pOwner)
      return false;
   auto pPartner = pOwner->GetNext(&track, false);
   if (!pPartner)
      return false;

   // Swap channels, avoiding copying of GroupData
   auto pData = move(track.mpGroupData);
   assert(pData);
   pOwner->MoveUp(pPartner);
   pPartner->mpGroupData = move(pData);
   pPartner->SetChannel(Track::LeftChannel);
   track.SetChannel(Track::RightChannel);
   return true;
}

void TrackList::Permute(const std::vector<TrackNodePointer> &permutation)
{
   for (const auto iter : permutation) {
      ListOfTracks::value_type track = *iter.first;
      erase(iter.first);
      Track *pTrack = track.get();
      pTrack->SetOwner(shared_from_this(),
         { insert(ListOfTracks::end(), track), this });
   }
   auto n = getBegin();
   RecalcPositions(n);
   PermutationEvent(n);
}

Track *TrackList::FindById( TrackId id )
{
   // Linear search.  Tracks in a project are usually very few.
   // Search only the non-pending tracks.
   auto it = std::find_if( ListOfTracks::begin(), ListOfTracks::end(),
      [=](const ListOfTracks::value_type &ptr){ return ptr->GetId() == id; } );
   if (it == ListOfTracks::end())
      return {};
   return it->get();
}

Track *TrackList::DoAddToHead(const std::shared_ptr<Track> &t)
{
   Track *pTrack = t.get();
   push_front(ListOfTracks::value_type(t));
   auto n = getBegin();
   pTrack->SetOwner(shared_from_this(), n);
   pTrack->SetId( TrackId{ ++sCounter } );
   RecalcPositions(n);
   AdditionEvent(n);
   return front().get();
}

Track *TrackList::DoAdd(const std::shared_ptr<Track> &t)
{
   push_back(t);

   auto n = getPrev( getEnd() );

   t->SetOwner(shared_from_this(), n);
   t->SetId( TrackId{ ++sCounter } );
   RecalcPositions(n);
   AdditionEvent(n);
   return back().get();
}

auto TrackList::Replace(Track * t, const ListOfTracks::value_type &with) ->
   ListOfTracks::value_type
{
   ListOfTracks::value_type holder;
   if (t && with) {
      auto node = t->GetNode();
      t->SetOwner({}, {});

      holder = *node.first;

      Track *pTrack = with.get();
      *node.first = with;
      pTrack->SetOwner(shared_from_this(), node);
      pTrack->SetId( t->GetId() );
      RecalcPositions(node);

      DeletionEvent(t->shared_from_this(), true);
      AdditionEvent(node);
   }
   return holder;
}

void TrackList::UnlinkChannels(Track& track)
{
   auto list = track.mList.lock();
   if (list.get() == this)
   {
      auto channels = TrackList::Channels(&track);
      for (auto c : channels)
      {
          c->SetLinkType(Track::LinkType::None);
          c->SetChannel(Track::ChannelType::MonoChannel);
      }
   }
   else
      THROW_INCONSISTENCY_EXCEPTION;
}

bool TrackList::MakeMultiChannelTrack(Track& track, int nChannels, bool aligned)
{
   if (nChannels != 2)
      return false;

   auto list = track.mList.lock();
   if (list.get() == this)
   {
      if (*list->FindLeader(&track) != &track)
         return false;

      auto first = list->Find(&track);
      auto canLink = [&]() -> bool {
         int count = nChannels;
         for (auto it = first, end = TrackList::end(); it != end && count; ++it)
         {
            if ((*it)->HasLinkedTrack())
               return false;
            --count;
         }
         return count == 0;
      }();

      if (!canLink)
         return false;

      (*first)->SetChannel(Track::LeftChannel);
      auto second = std::next(first);
      (*second)->SetChannel(Track::RightChannel);
      (*first)->SetLinkType(aligned ? Track::LinkType::Aligned : Track::LinkType::Group);
   }
   else
      THROW_INCONSISTENCY_EXCEPTION;
   return true;
}

TrackNodePointer TrackList::Remove(Track *t)
{
   auto result = getEnd();
   if (t) {
      auto node = t->GetNode();
      t->SetOwner({}, {});

      if ( !isNull( node ) ) {
         ListOfTracks::value_type holder = *node.first;

         result = getNext( node );
         erase(node.first);
         if ( !isNull( result ) )
            RecalcPositions(result);

         DeletionEvent(t->shared_from_this(), false);
      }
   }
   return result;
}

void TrackList::Clear(bool sendEvent)
{
   // Null out the back-pointers to this in tracks, in case there
   // are outstanding shared_ptrs to those tracks, making them outlive
   // the temporary ListOfTracks below.
   for ( auto pTrack: *this )
   {
      pTrack->SetOwner({}, {});
      
      if (sendEvent)
         DeletionEvent(pTrack->shared_from_this(), false);
   }
   
   for ( auto pTrack: mPendingUpdates )
   {
      pTrack->SetOwner({}, {});

      if (sendEvent)
         DeletionEvent(pTrack, false);
   }

   ListOfTracks tempList;
   tempList.swap( *this );

   ListOfTracks updating;
   updating.swap( mPendingUpdates );

   mUpdaters.clear();
}

/// Return a track in the list that comes after Track t
Track *TrackList::GetNext(Track * t, bool linked) const
{
   if (t) {
      auto node = t->GetNode();
      if ( !isNull( node ) ) {
         if ( linked && t->HasLinkedTrack() )
            node = getNext( node );

         if ( !isNull( node ) )
            node = getNext( node );

         if ( !isNull( node ) )
            return node.first->get();
      }
   }

   return nullptr;
}

Track *TrackList::GetPrev(Track * t, bool linked) const
{
   if (t) {
      TrackNodePointer prev;
      auto node = t->GetNode();
      if ( !isNull( node ) ) {
         // linked is true and input track second in team?
         if (linked) {
            prev = getPrev( node );
            if( !isNull( prev ) &&
                !t->HasLinkedTrack() && t->GetLinkedTrack() )
               // Make it the first
               node = prev;
         }

         prev = getPrev( node );
         if ( !isNull( prev ) ) {
            // Back up once
            node = prev;

            // Back up twice sometimes when linked is true
            if (linked) {
               prev = getPrev( node );
               if( !isNull( prev ) &&
                   !(*node.first)->HasLinkedTrack() && (*node.first)->GetLinkedTrack() )
                  node = prev;
            }

            return node.first->get();
         }
      }
   }

   return nullptr;
}

bool TrackList::CanMoveUp(Track * t) const
{
   return GetPrev(t, true) != NULL;
}

bool TrackList::CanMoveDown(Track * t) const
{
   return GetNext(t, true) != NULL;
}

// This is used when you want to swap the channel group starting
// at s1 with that starting at s2.
// The complication is that the tracks are stored in a single
// linked list.
void TrackList::SwapNodes(TrackNodePointer s1, TrackNodePointer s2)
{
   // if a null pointer is passed in, we want to know about it
   wxASSERT(!isNull(s1));
   wxASSERT(!isNull(s2));

   // Deal with first track in each team
   s1 = ( * FindLeader( s1.first->get() ) )->GetNode();
   s2 = ( * FindLeader( s2.first->get() ) )->GetNode();

   // Safety check...
   if (s1 == s2)
      return;

   // Be sure s1 is the earlier iterator
   if ((*s1.first)->GetIndex() >= (*s2.first)->GetIndex())
      std::swap(s1, s2);

   // For saving the removed tracks
   using Saved = std::vector< ListOfTracks::value_type >;
   Saved saved1, saved2;

   auto doSave = [&] ( Saved &saved, TrackNodePointer &s ) {
      size_t nn = Channels( s.first->get() ).size();
      saved.resize( nn );
      // Save them in backwards order
      while( nn-- )
         saved[nn] = *s.first, s.first = erase(s.first);
   };

   doSave( saved1, s1 );
   // The two ranges are assumed to be disjoint but might abut
   const bool same = (s1 == s2);
   doSave( saved2, s2 );
   if (same)
      // Careful, we invalidated s1 in the second doSave!
      s1 = s2;

   // Reinsert them
   auto doInsert = [&] ( Saved &saved, TrackNodePointer &s ) {
      Track *pTrack;
      for (auto & pointer : saved)
         pTrack = pointer.get(),
         // Insert before s, and reassign s to point at the new node before
         // old s; which is why we saved pointers in backwards order
         pTrack->SetOwner(shared_from_this(),
            s = { insert(s.first, pointer), this } );
   };
   // This does not invalidate s2 even when it equals s1:
   doInsert( saved2, s1 );
   // Even if s2 was same as s1, this correctly inserts the saved1 range
   // after the saved2 range, when done after:
   doInsert( saved1, s2 );

   // Now correct the Index in the tracks, and other things
   RecalcPositions(s1);
   PermutationEvent(s1);
}

bool TrackList::MoveUp(Track * t)
{
   if (t) {
      Track *p = GetPrev(t, true);
      if (p) {
         SwapNodes(p->GetNode(), t->GetNode());
         return true;
      }
   }

   return false;
}

bool TrackList::MoveDown(Track * t)
{
   if (t) {
      Track *n = GetNext(t, true);
      if (n) {
         SwapNodes(t->GetNode(), n->GetNode());
         return true;
      }
   }

   return false;
}

bool TrackList::Contains(const Track * t) const
{
   return make_iterator_range( *this ).contains( t );
}

bool TrackList::empty() const
{
   return begin() == end();
}

size_t TrackList::size() const
{
   int cnt = 0;

   if (!empty())
      cnt = getPrev( getEnd() ).first->get()->GetIndex() + 1;

   return cnt;
}

namespace {
   // Abstract the common pattern of the following three member functions
   inline double Accumulate
      (const TrackList &list,
       double (Track::*memfn)() const,
       double ident,
       const double &(*combine)(const double&, const double&))
   {
      // Default the answer to zero for empty list
      if (list.empty()) {
         return 0.0;
      }

      // Otherwise accumulate minimum or maximum of track values
      return list.Any().accumulate(ident, combine, memfn);
   }
}

double TrackList::GetMinOffset() const
{
   return Accumulate(*this, &Track::GetOffset, DBL_MAX, std::min);
}

double TrackList::GetStartTime() const
{
   return Accumulate(*this, &Track::GetStartTime, DBL_MAX, std::min);
}

double TrackList::GetEndTime() const
{
   return Accumulate(*this, &Track::GetEndTime, -DBL_MAX, std::max);
}

std::shared_ptr<Track>
TrackList::RegisterPendingChangedTrack( Updater updater, Track *src )
{
   std::shared_ptr<Track> pTrack;
   if (src) {
      pTrack = src->Clone(); // not duplicate
      // Share the satellites with the original, though they do not point back
      // to the pending track
      ((AttachedTrackObjects&)*pTrack) = *src; // shallow copy
   }

   if (pTrack) {
      mUpdaters.push_back( updater );
      mPendingUpdates.push_back( pTrack );
      auto n = mPendingUpdates.end();
      --n;
      pTrack->SetOwner(shared_from_this(), {n, &mPendingUpdates});
   }

   return pTrack;
}

void TrackList::RegisterPendingNewTrack( const std::shared_ptr<Track> &pTrack )
{
   Add<Track>( pTrack );
   pTrack->SetId( TrackId{} );
}

void TrackList::UpdatePendingTracks()
{
   auto pUpdater = mUpdaters.begin();
   for (const auto &pendingTrack : mPendingUpdates) {
      // Copy just a part of the track state, according to the update
      // function
      const auto &updater = *pUpdater;
      auto src = FindById( pendingTrack->GetId() );
      if (pendingTrack && src) {
         if (updater)
            updater( *pendingTrack, *src );
         pendingTrack->DoSetLinkType(src->GetLinkType());
      }
      ++pUpdater;
   }
}

/*! @excsafety{No-fail} */
void TrackList::ClearPendingTracks( ListOfTracks *pAdded )
{
   for (const auto &pTrack: mPendingUpdates)
      pTrack->SetOwner( {}, {} );
   mPendingUpdates.clear();
   mUpdaters.clear();

   if (pAdded)
      pAdded->clear();

   // To find the first node that remains after the first deleted one
   TrackNodePointer node;
   bool foundNode = false;

   for (auto it = ListOfTracks::begin(), stop = ListOfTracks::end();
        it != stop;) {
      if (it->get()->GetId() == TrackId{}) {
         do {
            if (pAdded)
               pAdded->push_back( *it );
            (*it)->SetOwner( {}, {} );
            DeletionEvent(*it, false);
            it = erase( it );
         }
         while (it != stop && it->get()->GetId() == TrackId{});

         if (!foundNode && it != stop) {
            node = (*it)->GetNode();
            foundNode = true;
         }
      }
      else
         ++it;
   }

   if (!empty()) {
      RecalcPositions(getBegin());
   }
}

/*! @excsafety{Strong} */
bool TrackList::ApplyPendingTracks()
{
   bool result = false;

   ListOfTracks additions;
   ListOfTracks updates;
   {
      // Always clear, even if one of the update functions throws
      auto cleanup = finally( [&] { ClearPendingTracks( &additions ); } );
      UpdatePendingTracks();
      updates.swap( mPendingUpdates );
   }

   // Remaining steps must be No-fail-guarantee so that this function
   // gives Strong-guarantee

   std::vector< std::shared_ptr<Track> > reinstated;

   for (auto &pendingTrack : updates) {
      if (pendingTrack) {
         pendingTrack->AttachedTrackObjects::ForEach([&](auto &attachment){
            attachment.Reparent( pendingTrack );
         });
         auto src = FindById( pendingTrack->GetId() );
         if (src)
            this->Replace(src, pendingTrack), result = true;
         else
            // Perhaps a track marked for pending changes got deleted by
            // some other action.  Recreate it so we don't lose the
            // accumulated changes.
            reinstated.push_back(pendingTrack);
      }
   }

   // If there are tracks to reinstate, append them to the list.
   for (auto &pendingTrack : reinstated)
      if (pendingTrack)
         this->Add( pendingTrack ), result = true;

   // Put the pending added tracks back into the list, preserving their
   // positions.
   bool inserted = false;
   ListOfTracks::iterator first;
   for (auto &pendingTrack : additions) {
      if (pendingTrack) {
         auto iter = ListOfTracks::begin();
         std::advance( iter, pendingTrack->GetIndex() );
         iter = ListOfTracks::insert( iter, pendingTrack );
         pendingTrack->SetOwner( shared_from_this(), {iter, this} );
         pendingTrack->SetId( TrackId{ ++sCounter } );
         if (!inserted) {
            first = iter;
            inserted = true;
         }
      }
   }
   if (inserted) {
      TrackNodePointer node{first, this};
      RecalcPositions(node);
      AdditionEvent(node);
      result = true;
   }

   return result;
}

std::shared_ptr<Track> Track::SubstitutePendingChangedTrack()
{
   // Linear search.  Tracks in a project are usually very few.
   auto pList = mList.lock();
   if (pList) {
      const auto id = GetId();
      const auto end = pList->mPendingUpdates.end();
      auto it = std::find_if(
         pList->mPendingUpdates.begin(), end,
         [=](const ListOfTracks::value_type &ptr){ return ptr->GetId() == id; } );
      if (it != end)
         return *it;
   }
   return SharedPointer();
}

std::shared_ptr<const Track> Track::SubstitutePendingChangedTrack() const
{
   return const_cast<Track*>(this)->SubstitutePendingChangedTrack();
}

std::shared_ptr<const Track> Track::SubstituteOriginalTrack() const
{
   auto pList = mList.lock();
   if (pList) {
      const auto id = GetId();
      const auto pred = [=]( const ListOfTracks::value_type &ptr ) {
         return ptr->GetId() == id; };
      const auto end = pList->mPendingUpdates.end();
      const auto it = std::find_if( pList->mPendingUpdates.begin(), end, pred );
      if (it != end) {
         const auto &list2 = (const ListOfTracks &) *pList;
         const auto end2 = list2.end();
         const auto it2 = std::find_if( list2.begin(), end2, pred );
         if ( it2 != end2 )
            return *it2;
      }
   }
   return SharedPointer();
}

auto Track::ClassTypeInfo() -> const TypeInfo &
{
   static Track::TypeInfo info{
      { "generic", "generic", XO("Generic Track") }, false };
   return info;
}

bool Track::SupportsBasicEditing() const
{
   return true;
}

auto Track::GetIntervals() const -> ConstIntervals
{
   return {};
}

auto Track::GetIntervals() -> Intervals
{
   return {};
}

// Serialize, not with tags of its own, but as attributes within a tag.
void Track::WriteCommonXMLAttributes(
   XMLWriter &xmlFile, bool includeNameAndSelected) const
{
   if (includeNameAndSelected) {
      xmlFile.WriteAttr(wxT("name"), GetName());
      xmlFile.WriteAttr(wxT("isSelected"), this->GetSelected());
   }
   AttachedTrackObjects::ForEach([&](auto &attachment){
      attachment.WriteXMLAttributes( xmlFile );
   });
}

// Return true iff the attribute is recognized.
bool Track::HandleCommonXMLAttribute(
   const std::string_view& attr, const XMLAttributeValueView& valueView)
{
   long nValue = -1;

   bool handled = false;
   AttachedTrackObjects::ForEach([&](auto &attachment){
      handled = handled || attachment.HandleXMLAttribute( attr, valueView );
   });
   if (handled)
      ;
   else if (attr == "name") {
      SetName(valueView.ToWString());
      return true;
   }
   else if (attr == "isSelected" && valueView.TryGet(nValue)) {
      this->SetSelected(nValue != 0);
      return true;
   }
   return false;
}

void Track::AdjustPositions()
{
   auto pList = mList.lock();
   if (pList) {
      pList->RecalcPositions(mNode);
      pList->ResizingEvent(mNode);
   }
}

auto AudioTrack::ClassTypeInfo() -> const TypeInfo &
{
   static Track::TypeInfo info{
      { "audio", "audio", XO("Audio Track") },
      false, &Track::ClassTypeInfo() };
   return info;
}

auto PlayableTrack::ClassTypeInfo() -> const TypeInfo &
{
   static Track::TypeInfo info{
      { "playable", "playable", XO("Playable Track") },
      false, &AudioTrack::ClassTypeInfo() };
   return info;
}

TrackIntervalData::~TrackIntervalData() = default;

bool TrackList::HasPendingTracks() const
{
   if ( !mPendingUpdates.empty() )
      return true;
   if (end() != std::find_if(begin(), end(), [](const Track *t){
      return t->GetId() == TrackId{};
   }))
      return true;
   return false;
}

Track::LinkType Track::GetLinkType() const noexcept
{
    return mpGroupData ? mpGroupData->mLinkType : LinkType::None;
}

bool Track::IsAlignedWithLeader() const
{
   if (auto owner = GetOwner())
   {
      auto leader = *owner->FindLeader(this);
      return leader != this && leader->GetLinkType() == Track::LinkType::Aligned;
   }
   return false;
}
