//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// BESTtoolbox.h --------------------------------------------------------------
//=================================================================================
// Header file containing functions for use with CMS EDAnalyzer and EDProducer ----
///////////////////////////////////////////////////////////////////////////////////

// make sure the functions are not declared more than once
#ifndef BESTtoolbox_H
#define BESTtoolbox_H

// include files
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"
#include "TMath.h"
#include "TLorentzVector.h"

// Fast Jet Include files
#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include "fastjet/tools/Filter.hh"
#include <fastjet/ClusterSequence.hh>
#include <fastjet/ActiveAreaSpec.hh>
#include <fastjet/ClusterSequenceArea.hh>

///////////////////////////////////////////////////////////////////////////////////
// Functions ----------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

// calculate Legendre Polynomials
float LegendreP(float x, int order);

// calculate Fox Wolfram moments
int FWMoments(std::vector<TLorentzVector> particles, double (&outputs)[5] );

// get jet's constituents
void getJetDaughters(std::vector<reco::Candidate * > &daughtersOfJet, std::vector<pat::Jet>::const_iterator jet);

// store the jet variables
void storeJetVariables(std::map<std::string, float> &besVars, std::vector<pat::Jet>::const_iterator jet, int jetColl);

// store the secondary vertex variables
void storeSecVertexVariables(std::map<std::string, float> &besVars, std::map< std::string, std::vector<float> > &jetVecVars,
                             TLorentzVector jet, std::vector<reco::VertexCompositePtrCandidate> secVertices);

// calculate the rest frame variables
bool calcBESvariables(std::map<std::string, float> &besVars, std::vector<reco::Candidate *> &daughtersOfJet,
                      std::map<std::string, std::vector<TLorentzVector> > &boostedDaughters,
                      std::vector<pat::Jet>::const_iterator jet, std::map<std::string, std::vector<fastjet::PseudoJet> > &restJets,
                    //   int mass);
                      std::string mass);

// store the daughters, rest frame daughters, and rest frame jets
void storeJetDaughters(std::vector<reco::Candidate * > &daughtersOfJet, std::vector<pat::Jet>::const_iterator jet,
                       std::map<std::string, std::vector<TLorentzVector> > &boostedDaughters,
                    //    std::map<std::string, std::vector<fastjet::PseudoJet> > &restJets, std::vector<int> restMasses,
                       std::map<std::string, std::vector<fastjet::PseudoJet> > &restJets, std::vector<std::string> restMasses,
                       std::map<std::string, std::vector<float> > &jetVecVars, int jetColl );

// make rest frame z axis the boost axis
void pboost( TVector3 pbeam, TVector3 plab, TLorentzVector &pboo );

#endif
