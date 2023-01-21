/**********************************************************************
 
 Audacity: A Digital Audio Editor
 
 RealtimeEffectManager.cpp
 
 Paul Licameli split from EffectManager.cpp
 
 **********************************************************************/


#include "RealtimeEffectManager.h"
#include "RealtimeEffectState.h"

#include <memory>
#include "Project.h"
#include "Track.h"

#include <atomic>
#include <wx/time.h>

static const AttachedProjectObjects::RegisteredFactory manager
{
   [](AudacityProject &project)
   {
      return std::make_shared<RealtimeEffectManager>(project);
   }
};

RealtimeEffectManager &RealtimeEffectManager::Get(AudacityProject &project)
{
   return project.AttachedObjects::Get<RealtimeEffectManager&>(manager);
}

const RealtimeEffectManager &RealtimeEffectManager::Get(const AudacityProject &project)
{
   return Get(const_cast<AudacityProject &>(project));
}

RealtimeEffectManager::RealtimeEffectManager(AudacityProject &project)
   : mProject(project)
{
}

RealtimeEffectManager::~RealtimeEffectManager()
{
}

bool RealtimeEffectManager::IsActive() const noexcept
{
   return mActive;
}

void RealtimeEffectManager::Initialize(
   RealtimeEffects::InitializationScope &scope, double sampleRate)
{
   // (Re)Set processor parameters
   mRates.clear();
   mGroupLeaders.clear();

   // RealtimeAdd/RemoveEffect() needs to know when we're active so it can
   // initialize newly added effects
   mActive = true;

   // Tell each state to get ready for action
   VisitAll([&scope, sampleRate](RealtimeEffectState &state, bool) {
      scope.mInstances.push_back(state.Initialize(sampleRate));
   });

   // Leave suspended state
   SetSuspended(false);
}

void RealtimeEffectManager::AddTrack(
   RealtimeEffects::InitializationScope &scope,
   Track &track, unsigned chans, float rate)
{
   auto leader = *track.GetOwner()->FindLeader(&track);
   // This should never return a null
   wxASSERT(leader);
   mGroupLeaders.push_back(leader);
   mRates.insert({leader, rate});

   VisitGroup(*leader,
      [&](RealtimeEffectState & state, bool) {
         scope.mInstances.push_back(state.AddTrack(*leader, chans, rate));
      }
   );
}

void RealtimeEffectManager::Finalize() noexcept
{
   // Reenter suspended state
   SetSuspended(true);

   // Assume it is now safe to clean up
   mLatency = std::chrono::microseconds(0);

   VisitAll([](RealtimeEffectState &state, bool){ state.Finalize(); });

   // Reset processor parameters
   mGroupLeaders.clear();
   mRates.clear();

   // No longer active
   mActive = false;
}

//
// This will be called in a different thread than the main GUI thread.
//
void RealtimeEffectManager::ProcessStart(bool suspended)
{
   // Can be suspended because of the audio stream being paused or because effects
   // have been suspended.
   VisitAll([suspended](RealtimeEffectState &state, bool listIsActive){
      state.ProcessStart(!suspended && listIsActive);
   });
}

//

// This will be called in a thread other than the main GUI thread.
//
size_t RealtimeEffectManager::Process(bool suspended, Track &track,
   float *const *buffers, float *const *scratch, float *const dummy,
   unsigned nBuffers, size_t numSamples)
{
   // Can be suspended because of the audio stream being paused or because effects
   // have been suspended, so allow the samples to pass as-is.
   if (suspended)
      return 0;

   // Remember when we started so we can calculate the amount of latency we
   // are introducing
   auto start = std::chrono::steady_clock::now();

   // Allocate the in and out buffer arrays
   const auto ibuf =
      static_cast<float **>(alloca(nBuffers * sizeof(float *)));
   const auto obuf =
      static_cast<float **>(alloca(nBuffers * sizeof(float *)));

   // And populate the input with the buffers we've been given while allocating
   // NEW output buffers
   for (unsigned int i = 0; i < nBuffers; i++) {
      ibuf[i] = buffers[i];
      obuf[i] = scratch[i];
   }

   // Now call each effect in the chain while swapping buffer pointers to feed the
   // output of one effect as the input to the next effect
   // Tracks how many processors were called
   size_t called = 0;
   size_t discardable = 0;
   VisitGroup(track,
      [&](RealtimeEffectState &state, bool)
      {
         discardable +=
            state.Process(track, nBuffers, ibuf, obuf, dummy, numSamples);
         for (auto i = 0; i < nBuffers; ++i)
            std::swap(ibuf[i], obuf[i]);
         called++;
      }
   );

   // Once we're done, we might wind up with the last effect storing its results
   // in the temporary buffers.  If that's the case, we need to copy it over to
   // the caller's buffers.  This happens when the number of effects processed
   // is odd.
   if (called & 1)
      for (unsigned int i = 0; i < nBuffers; i++)
         memcpy(buffers[i], ibuf[i], numSamples * sizeof(float));

   // Remember the latency
   auto end = std::chrono::steady_clock::now();
   mLatency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

   //
   // This is wrong...needs to handle tails
   //
   return discardable;
}

//
// This will be called in a different thread than the main GUI thread.
//
void RealtimeEffectManager::ProcessEnd(bool suspended) noexcept
{
   // Can be suspended because of the audio stream being paused or because effects
   // have been suspended.
   VisitAll([suspended](RealtimeEffectState &state, bool){
      state.ProcessEnd();
   });
}

RealtimeEffectManager::
AllListsLock::AllListsLock(RealtimeEffectManager *pManager)
   : mpManager{ pManager }
{
   if (mpManager) {
      // Paralleling VisitAll
      RealtimeEffectList::Get(mpManager->mProject).GetLock().lock();
      // And all track lists
      for (auto leader : mpManager->mGroupLeaders)
         RealtimeEffectList::Get(*leader).GetLock().lock();
   }
}

RealtimeEffectManager::AllListsLock::AllListsLock(AllListsLock &&other)
   : mpManager{ other.mpManager }
{
   other.mpManager = nullptr;
}

RealtimeEffectManager::AllListsLock&
RealtimeEffectManager::AllListsLock::operator =(AllListsLock &&other)
{
   if (this != &other) {
      Reset();
      mpManager = other.mpManager;
      other.mpManager = nullptr;
   }
   return *this;
}

void RealtimeEffectManager::AllListsLock::Reset()
{
   if (mpManager) {
      // Paralleling VisitAll
      RealtimeEffectList::Get(mpManager->mProject).GetLock().unlock();
      // And all track lists
      for (auto leader : mpManager->mGroupLeaders)
         RealtimeEffectList::Get(*leader).GetLock().unlock();
      mpManager = nullptr;
   }
}

std::shared_ptr<RealtimeEffectState>
RealtimeEffectManager::MakeNewState(
   RealtimeEffects::InitializationScope *pScope,
   Track *pLeader, const PluginID &id)
{
   if (!pScope && mActive)
      return nullptr;
   auto pNewState = RealtimeEffectState::make_shared(id);
   auto &state = *pNewState;
   if (pScope && mActive) {
      // Adding a state while playback is in-flight
      auto pInstance = state.Initialize(pScope->mSampleRate);
      pScope->mInstances.push_back(pInstance);
      for (auto &leader : mGroupLeaders) {
         // Add all tracks to a per-project state, but add only the same track
         // to a state in the per-track list
         if (pLeader && pLeader != leader)
            continue;
         auto rate = mRates[leader];
         auto pInstance2 =
            state.AddTrack(*leader, pScope->mNumPlaybackChannels, rate);
         if (pInstance2 != pInstance)
            pScope->mInstances.push_back(pInstance2);
      }
   }
   return pNewState;
}

namespace {
std::pair<Track *, RealtimeEffectList &>
FindStates(AudacityProject &project, Track *pTrack) {
   auto pLeader = pTrack ? *TrackList::Channels(pTrack).begin() : nullptr;
   return { pLeader,
      pLeader
         ? RealtimeEffectList::Get(*pLeader)
         : RealtimeEffectList::Get(project)
   };
}
}

std::shared_ptr<RealtimeEffectState> RealtimeEffectManager::AddState(
   RealtimeEffects::InitializationScope *pScope,
   Track *pTrack, const PluginID & id)
{
   auto [pLeader, states] = FindStates(mProject, pTrack);
   auto pState = MakeNewState(pScope, pTrack, id);
   if (!pState)
      return nullptr;

   // Only now add the completed state to the list, under a lock guard
   if (!states.AddState(pState))
      return nullptr;
   Publish({
      RealtimeEffectManagerMessage::Type::EffectAdded,
      pLeader ? pLeader->shared_from_this() : nullptr
   });
   return pState;
}

std::shared_ptr<RealtimeEffectState> RealtimeEffectManager::ReplaceState(
   RealtimeEffects::InitializationScope *pScope,
   Track *pTrack, size_t index, const PluginID & id)
{
   auto [pLeader, states] = FindStates(mProject, pTrack);
   auto pOldState = states.GetStateAt(index);
   if (!pOldState)
      return nullptr;
   auto pNewState = MakeNewState(pScope, pTrack, id);
   if (!pNewState)
      return nullptr;

   // Only now swap the completed state into the list, under a lock guard
   if (!states.ReplaceState(index, pNewState))
      return nullptr;
   if (mActive)
      pOldState->Finalize();
   Publish({
      RealtimeEffectManagerMessage::Type::EffectReplaced,
      pLeader ? pLeader->shared_from_this() : nullptr
   });
   return pNewState;
}

void RealtimeEffectManager::RemoveState(
   RealtimeEffects::InitializationScope *pScope,
   Track *pTrack, const std::shared_ptr<RealtimeEffectState> pState)
{
   auto [pLeader, states] = FindStates(mProject, pTrack);

   // Remove the state from processing (under the lock guard) before finalizing
   states.RemoveState(pState);
   if (mActive)
      pState->Finalize();
   Publish({
      RealtimeEffectManagerMessage::Type::EffectRemoved,
      pLeader ? pLeader->shared_from_this() : nullptr
   });
}

std::optional<size_t> RealtimeEffectManager::FindState(
   Track *pTrack, const std::shared_ptr<RealtimeEffectState> &pState) const
{
   auto [_, states] = FindStates(mProject, pTrack);
   return states.FindState(pState);
}

// Where is this used?
#if 0
auto RealtimeEffectManager::GetLatency() const -> Latency
{
   return mLatency; // should this be atomic?
}
#endif
