/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "MySynthesiser.h"

//==============================================================================
/**
*/
class KarPlusPlus2AudioProcessor : public foleys::MagicProcessor
{
public:
    //==============================================================================
    KarPlusPlus2AudioProcessor();
    ~KarPlusPlus2AudioProcessor();

    //==============================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
#endif

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
//    juce::AudioProcessorEditor* createEditor() override;
//    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    //==============================================================================
//    void getStateInformation(juce::MemoryBlock& destData) override;
//    void setStateInformation(const void* data, int sizeInBytes) override;

    //==============================================================================
    juce::AudioProcessorValueTreeState apvts; // Needs to be public
    
    
private:
    juce::AudioProcessorValueTreeState::ParameterLayout createParams();
    juce::AudioProcessorValueTreeState::ParameterLayout addVelToParams();
    
    foleys::MagicPlotSource* analyser = nullptr;
    
    juce::Synthesiser synth;
    int voiceCount = 12;
    
//    foleys::MagicProcessorState magicState { *this, apvts };
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(KarPlusPlus2AudioProcessor)
};
