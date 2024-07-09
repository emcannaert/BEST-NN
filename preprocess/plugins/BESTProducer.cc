// -*- C++ -*-
//========================================================================================
// Package: BEST/preprocess           ---------------------------------------
// Class:     BESTProducer            ---------------------------------------
//----------------------------------------------------------------------------------------
/**\class BESTProducer BESTProducer.cc BEST/preprocess/plugins/BESTProducer.cc
------------------------------------------------------------------------------------------
 Description: This class preprocesses MC samples so that they can be used with BEST ---
 -----------------------------------------------------------------------------------------
 Implementation:                                    ---
    This EDProducer is meant to be used with CMSSW_9_4_8                 ---
*/
//========================================================================================
// Authors:  Brendan Regnery, Samantha Abbott, Justin Pilot, Reyer Band, Devin Taylor ---------------------
//     Created:  WED, 8 Aug 2018 21:00:28 GMT  ---------------------------------------
//   Modified and adapted by Ethan Cannaert, 2019-2024 
//========================================================================================
//////////////////////////////////////////////////////////////////////////////////////////


// system include files
#include <memory>
#include <thread>
#include <iostream>
#include<TRandom3.h>

// FWCore include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// Data Formats and tools include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"

// Fast Jet Include files
#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include "fastjet/tools/Filter.hh"
#include <fastjet/ClusterSequence.hh>
#include <fastjet/ActiveAreaSpec.hh>
#include <fastjet/ClusterSequenceArea.hh>

#include "DataFormats/Candidate/interface/Candidate.h"

//include jet sorting utility
#include "sortJets.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
// ROOT include files
#include "TTree.h"
#include "TFile.h"
#include "TH2F.h"
#include "TLorentzVector.h"
#include "TCanvas.h"

// user made files
#include "BESTtoolbox.h"


#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetResolutionObject.h"
#include "JetMETCorrections/Modules/interface/JetResolution.h"
#include "PhysicsTools/PatUtils/interface/SmearedJetProducerT.h"

#include "CondFormats/DataRecord/interface/JetResolutionRcd.h"
#include "CondFormats/DataRecord/interface/JetResolutionScaleFactorRcd.h"

typedef math::XYZTLorentzVector LorentzVector;

typedef reco::Candidate::PolarLorentzVector fourv;
typedef math::XYZVector Vector;
using namespace reco;


class BESTProducer : public edm::stream::EDProducer<> 
{
   public:
     explicit BESTProducer(const edm::ParameterSet&);
     ~BESTProducer();
     bool isMatchedtoSJ(std::vector<TLorentzVector>, TLorentzVector);
     static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
     double calcMPP(TLorentzVector );
     const reco::Candidate* parse_chain(const reco::Candidate*);
     bool isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF, bool jetPUid, const float iJet_pt);
     bool isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF, int nfatjets);
     bool isHEM(const float jet_eta, const float jet_phi);

     //===========================================================================
     // User functions -----------------------------------------------------------
     //===========================================================================

   private:
     virtual void beginStream(edm::StreamID) override;
     virtual void produce(edm::Event&, const edm::EventSetup&) override;
     virtual void endStream() override;

     //===========================================================================
     // Member Data --------------------------------------------------------------
     //===========================================================================

     // Tree variables
     TTree *superjetTree;
     std::map<std::string, float> treeVars;
     std::vector<std::string> listOfVars;

      edm::FileInPath JECUncert_AK8_path;
      edm::FileInPath JECUncert_AK4_path;
      edm::EDGetTokenT<double> m_rho_token;

      std::string year;
      std::string lumiTag;


     int nEvents = 0;
     int nWbHt = 0;
     int nWbZt = 0;
     int nWbWb = 0;
     int nHtZt = 0;
     int nHtHt = 0;
     int nZtZt = 0;
     int nPassedEvents = 0;
     bool testNewVars = true;

      TRandom3 *randomNum = new TRandom3(); // for JERs

     // Tokens
     edm::EDGetTokenT<std::vector<pat::Jet>> jetToken_;
     edm::EDGetTokenT<std::vector<pat::Jet> > fatJetToken_;

     edm::EDGetTokenT<std::vector<reco::GenParticle> > genPartToken_;
     std::string jetType_;

};


bool BESTProducer::isMatchedtoSJ(std::vector<TLorentzVector> superJetTLVs, TLorentzVector candJet)
{
   for(auto iJet = superJetTLVs.begin(); iJet!=superJetTLVs.end(); iJet++)
   {
     if( abs(candJet.Angle(iJet->Vect())) < 0.001) return true;
   }
   return false;
}

double BESTProducer::calcMPP(TLorentzVector superJetTLV ) 
{
   TVector3 jet_axis(superJetTLV.Px()/superJetTLV.P(),superJetTLV.Py()/superJetTLV.P(),superJetTLV.Pz()/superJetTLV.P());
   double min_pp = 99999999999999.;
   double min_boost = 0.;

   for(int iii=0;iii<10000;iii++)
   {
     TLorentzVector superJetTLV_ = superJetTLV;
     double beta_cand = iii/10000.;
     superJetTLV_.Boost(-beta_cand*jet_axis.X(),-beta_cand*jet_axis.Y(),-beta_cand*jet_axis.Z());
     if(abs( ( superJetTLV_.Px()*superJetTLV.Px()+superJetTLV_.Py()*superJetTLV.Py() +superJetTLV_.Pz()*superJetTLV.Pz() )/superJetTLV.P() ) < min_pp) 
      {
      min_boost = beta_cand; 
      min_pp = abs( ( superJetTLV_.Px()*superJetTLV.Px()+superJetTLV_.Py()*superJetTLV.Py() +superJetTLV_.Pz()*superJetTLV.Pz() )/superJetTLV.P() ) ;
      }
   }
   return min_boost;
}
bool BESTProducer::isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF,bool jetPUid, const float iJet_pt)
{
   if( (abs(eta) > 2.4)) return false;


   // apply the MEDIUM PU jet id https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
   if( (!jetPUid) && (iJet_pt < 50.0)) return false;

   if ((NHF>0.9) || (NEMF>0.9) || (NumConst<1) || (CHF<0.) || (CHM<0) || (MUF > 0.8) || (CEMF > 0.8)) 
      {
         return false;
      }
   else{ return true;}

}
// checks jet (tight) ID
bool BESTProducer::isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF, int nfatjets)
{
   if ( (nfatjets < 2) && (abs(eta) > 2.4) ) return false;
   else if ( (nfatjets >= 2) && (abs(eta) > 1.5) ) return false;

   if ((NHF>0.9) || (NEMF>0.9) || (NumConst<1) || (CHF<0.) || (CHM<0) || (MUF > 0.8) || (CEMF > 0.8)) 
      {
         return false;
      }
   else{ return true;}

}
const reco::Candidate* BESTProducer::parse_chain(const reco::Candidate* cand)
{  
   for (unsigned int iii=0; iii<cand->numberOfDaughters(); iii++)
   {
     if(cand->daughter(iii)->pdgId() == cand->pdgId()) return parse_chain(cand->daughter(iii));
   }
   return cand;
}
bool BESTProducer::isHEM(const float jet_eta, const float jet_phi)
{

   if(year != "2018")return false; // HEM is only relevant for 2018
   if( (jet_phi >  -1.57)&&( jet_phi < -0.87) )
   {
      if( (jet_eta > -3.0)&&(jet_eta < -1.3))return true;

   }
   return false;
}
//given a list of (boosted to specific SJ COM) particles coming from b-tagged AK8 subjets and the particles of a specific AK4 jet
//returns fraction of those AK4 jet particles that originated from B-tagged subjets
// returns 0 if num AK4 daughters < 5


BESTProducer::BESTProducer(const edm::ParameterSet& iConfig)
{
   

   edm::Service<TFileService> fs;
   superjetTree = fs->make<TTree>("superjetTree","superjetTree");
   jetType_ = iConfig.getParameter<std::string>("jetType");


   //------------------------------------------------------------------------------
   // Create tree variables and branches ------------------------------------------
   //------------------------------------------------------------------------------
   // listOfVars is the flat part of the TTree ------------------------------------
   //------------------------------------------------------------------------------

     // SJ COM variable information
     listOfVars.push_back("tot_HT");
     listOfVars.push_back("eventNumber");

     //SJ mass variables
     listOfVars.push_back("SJ_mass");
     listOfVars.push_back("SJ_mass_25");
     listOfVars.push_back("SJ_mass_50");
     listOfVars.push_back("SJ_mass_100");
     listOfVars.push_back("SJ_mass_150");
     listOfVars.push_back("SJ_mass_200");
     listOfVars.push_back("SJ_mass_300");

   
     //SJ nAK4 variables
     listOfVars.push_back("SJ_nAK4_25");
     listOfVars.push_back("SJ_nAK4_50");
     listOfVars.push_back("SJ_nAK4_100");
     listOfVars.push_back("SJ_nAK4_150");
     listOfVars.push_back("SJ_nAK4_200");
     listOfVars.push_back("SJ_nAK4_300");

     listOfVars.push_back("AK4_m1");  
     listOfVars.push_back("AK4_m2");
     listOfVars.push_back("AK4_m3");
     listOfVars.push_back("AK4_m4");

     listOfVars.push_back("AK4_m12");  
     listOfVars.push_back("AK4_m13");  
     listOfVars.push_back("AK4_m14");  
     listOfVars.push_back("AK4_m23");  
     listOfVars.push_back("AK4_m24");  
     listOfVars.push_back("AK4_m34");  

     listOfVars.push_back("AK4_m123");  
     listOfVars.push_back("AK4_m124");  
     listOfVars.push_back("AK4_m134");  
     listOfVars.push_back("AK4_m234");  

     listOfVars.push_back("AK4_m1234");  


     listOfVars.push_back("AK41_E");  
     listOfVars.push_back("AK42_E");  
     listOfVars.push_back("AK43_E");  
     listOfVars.push_back("AK44_E");  

     //reclustered AK4 jet angles
     listOfVars.push_back("AK4_theta12");   
     listOfVars.push_back("AK4_theta13");
     listOfVars.push_back("AK4_theta14");
     listOfVars.push_back("AK4_theta23");
     listOfVars.push_back("AK4_theta24");
     listOfVars.push_back("AK4_theta34");

     //AK4 jet boosted information - boost reclustered AK4 jets into their COM and look at BES variables, ndaughters, nsubjettiness
     //listOfVars.push_back("AK41_Tau3");
     //listOfVars.push_back("AK41_Tau2");
     //listOfVars.push_back("AK41_Tau1");
     //listOfVars.push_back("AK41_Tau21");
     listOfVars.push_back("AK41_nsubjets");
     listOfVars.push_back("AK41_thrust");
     listOfVars.push_back("AK41_sphericity");
     listOfVars.push_back("AK41_asymmetry");
     listOfVars.push_back("AK41_isotropy");
     listOfVars.push_back("AK41_aplanarity");
     listOfVars.push_back("AK41_FW1");
     listOfVars.push_back("AK41_FW2");
     listOfVars.push_back("AK41_FW3");
     listOfVars.push_back("AK41_FW4");

     //listOfVars.push_back("AK42_Tau3");
     //listOfVars.push_back("AK42_Tau2");
     //listOfVars.push_back("AK42_Tau1");
     //listOfVars.push_back("AK42_Tau21");
     listOfVars.push_back("AK42_nsubjets");
     listOfVars.push_back("AK42_thrust");
     listOfVars.push_back("AK42_sphericity");
     listOfVars.push_back("AK42_asymmetry");
     listOfVars.push_back("AK42_isotropy");
     listOfVars.push_back("AK42_aplanarity");
     listOfVars.push_back("AK42_FW1");
     listOfVars.push_back("AK42_FW2");
     listOfVars.push_back("AK42_FW3");
     listOfVars.push_back("AK42_FW4");

     //listOfVars.push_back("AK43_Tau3");
     //listOfVars.push_back("AK43_Tau2");
     //listOfVars.push_back("AK43_Tau1");
     //listOfVars.push_back("AK43_Tau21");
     listOfVars.push_back("AK43_nsubjets");
     listOfVars.push_back("AK43_thrust");
     listOfVars.push_back("AK43_sphericity");
     listOfVars.push_back("AK43_asymmetry");
     listOfVars.push_back("AK43_isotropy");
     listOfVars.push_back("AK43_aplanarity");
     listOfVars.push_back("AK43_FW1");
     listOfVars.push_back("AK43_FW2");
     listOfVars.push_back("AK43_FW3");
     listOfVars.push_back("AK43_FW4");

     //listOfVars.push_back("AK43_Tau3");
     //listOfVars.push_back("AK43_Tau2");
     //listOfVars.push_back("AK43_Tau1");
     //listOfVars.push_back("AK43_Tau21");
     listOfVars.push_back("AK44_nsubjets");
     listOfVars.push_back("AK44_thrust");
     listOfVars.push_back("AK44_sphericity");
     listOfVars.push_back("AK44_asymmetry");
     listOfVars.push_back("AK44_isotropy");
     listOfVars.push_back("AK44_aplanarity");
     listOfVars.push_back("AK44_FW1");
     listOfVars.push_back("AK44_FW2");
     listOfVars.push_back("AK44_FW3");
     listOfVars.push_back("AK44_FW4");

     //SJ BES variables
     listOfVars.push_back("SJ_thrust");
     listOfVars.push_back("SJ_sphericity");
     listOfVars.push_back("SJ_asymmetry");
     listOfVars.push_back("SJ_isotropy");
     listOfVars.push_back("SJ_aplanarity");
     listOfVars.push_back("SJ_FW1");
     listOfVars.push_back("SJ_FW2");
     listOfVars.push_back("SJ_FW3");
     listOfVars.push_back("SJ_FW4");

     // new vars I added
     listOfVars.push_back("AK41_nDaughters");
     listOfVars.push_back("AK42_nDaughters");
     listOfVars.push_back("AK43_nDaughters");
     listOfVars.push_back("AK44_nDaughters");

     listOfVars.push_back("SJ_mass_400");
     listOfVars.push_back("SJ_mass_500");
     listOfVars.push_back("SJ_mass_800");
     listOfVars.push_back("SJ_mass_1000");

     listOfVars.push_back("AK41_px");
     listOfVars.push_back("AK42_px");
     listOfVars.push_back("AK43_px");
     listOfVars.push_back("AK44_px");

     listOfVars.push_back("AK41_py");
     listOfVars.push_back("AK42_py");
     listOfVars.push_back("AK43_py");
     listOfVars.push_back("AK44_py");

     listOfVars.push_back("AK41_pz");
     listOfVars.push_back("AK42_pz");
     listOfVars.push_back("AK43_pz");
     listOfVars.push_back("AK44_pz");

     listOfVars.push_back("SJ_nAK4_400");
     listOfVars.push_back("SJ_nAK4_500");
     listOfVars.push_back("SJ_nAK4_800");
     listOfVars.push_back("SJ_nAK4_1000");

   if(testNewVars)
   {
      /*
      // fraction of daughters that have energy greater than threshold
      AK41_daughters_frac_10

      AK41_mass_10   /// mass of jet using only daughters with energy greater than 10 GeV
      */

        listOfVars.push_back("SJ_AK4_frac_10");
        listOfVars.push_back("SJ_AK4_frac_25");
        listOfVars.push_back("SJ_AK4_frac_50");
        listOfVars.push_back("SJ_AK4_frac_75");
        listOfVars.push_back("SJ_AK4_frac_100");
        listOfVars.push_back("SJ_AK4_frac_200");
        listOfVars.push_back("SJ_AK4_frac_300");
        listOfVars.push_back("SJ_AK4_frac_500");
        listOfVars.push_back("SJ_AK4_frac_800");


        // AK41 vars

        listOfVars.push_back("AK41_daughters_frac_0p1");
        listOfVars.push_back("AK41_daughters_frac_0p5");
        listOfVars.push_back("AK41_daughters_frac_1");
        listOfVars.push_back("AK41_daughters_frac_2");
        listOfVars.push_back("AK41_daughters_frac_5");
        listOfVars.push_back("AK41_daughters_frac_7p5");
        listOfVars.push_back("AK41_daughters_frac_10");
        listOfVars.push_back("AK41_daughters_frac_15");

        //listOfVars.push_back("AK41_daughters_frac_20");
        //listOfVars.push_back("AK41_daughters_frac_40");
        //listOfVars.push_back("AK41_daughters_frac_50");
        //listOfVars.push_back("AK41_daughters_frac_75");
        //listOfVars.push_back("AK41_daughters_frac_100");


        listOfVars.push_back("AK41_mass_0p1");
        listOfVars.push_back("AK41_mass_0p5");
        listOfVars.push_back("AK41_mass_1");
        listOfVars.push_back("AK41_mass_2");
        listOfVars.push_back("AK41_mass_7p5");
        listOfVars.push_back("AK41_mass_10");
        listOfVars.push_back("AK41_mass_15");

        //listOfVars.push_back("AK41_mass_20");
        //listOfVars.push_back("AK41_mass_40");
        //listOfVars.push_back("AK41_mass_50");
        //listOfVars.push_back("AK41_mass_75");
        //listOfVars.push_back("AK41_mass_100");






        // AK42 vars
        listOfVars.push_back("AK42_daughters_frac_0p1");
        listOfVars.push_back("AK42_daughters_frac_0p5");
        listOfVars.push_back("AK42_daughters_frac_1");
        listOfVars.push_back("AK42_daughters_frac_2");
        listOfVars.push_back("AK42_daughters_frac_5");
        listOfVars.push_back("AK42_daughters_frac_7p5");
        listOfVars.push_back("AK42_daughters_frac_10");
        listOfVars.push_back("AK42_daughters_frac_15");

        listOfVars.push_back("AK42_mass_0p1");
        listOfVars.push_back("AK42_mass_0p5");
        listOfVars.push_back("AK42_mass_1");
        listOfVars.push_back("AK42_mass_2");
        listOfVars.push_back("AK42_mass_7p5");
        listOfVars.push_back("AK42_mass_10");
        listOfVars.push_back("AK42_mass_15");

        // AK43 vars

        listOfVars.push_back("AK43_daughters_frac_0p1");
        listOfVars.push_back("AK43_daughters_frac_0p5");
        listOfVars.push_back("AK43_daughters_frac_1");
        listOfVars.push_back("AK43_daughters_frac_2");
        listOfVars.push_back("AK43_daughters_frac_5");
        listOfVars.push_back("AK43_daughters_frac_7p5");
        listOfVars.push_back("AK43_daughters_frac_10");
        listOfVars.push_back("AK43_daughters_frac_15");

        listOfVars.push_back("AK43_mass_0p1");
        listOfVars.push_back("AK43_mass_0p5");
        listOfVars.push_back("AK43_mass_1");
        listOfVars.push_back("AK43_mass_2");
        listOfVars.push_back("AK43_mass_7p5");
        listOfVars.push_back("AK43_mass_10");
        listOfVars.push_back("AK43_mass_15");

        listOfVars.push_back("AK44_daughters_frac_0p1");
        listOfVars.push_back("AK44_daughters_frac_0p5");
        listOfVars.push_back("AK44_daughters_frac_1");
        listOfVars.push_back("AK44_daughters_frac_2");
        listOfVars.push_back("AK44_daughters_frac_5");
        listOfVars.push_back("AK44_daughters_frac_7p5");
        listOfVars.push_back("AK44_daughters_frac_10");
        listOfVars.push_back("AK44_daughters_frac_15");

        listOfVars.push_back("AK44_mass_0p1");
        listOfVars.push_back("AK44_mass_0p5");
        listOfVars.push_back("AK44_mass_1");
        listOfVars.push_back("AK44_mass_2");
        listOfVars.push_back("AK44_mass_7p5");
        listOfVars.push_back("AK44_mass_10");
        listOfVars.push_back("AK44_mass_15");


   }

     // need to add more variables here ...



   // init listofVars
   for (unsigned i = 0; i < listOfVars.size(); i++)
   {
     treeVars[ listOfVars[i] ] = -999.99;
     superjetTree->Branch( (listOfVars[i]).c_str() , &(treeVars[ listOfVars[i] ]), (listOfVars[i]+"/F").c_str() );
   }

   // import in settings for event 
   year      = iConfig.getParameter<std::string>("year");

   // Gen Particles
   genPartToken_ = consumes<std::vector<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("genPartCollection"));

   // jets
   jetToken_  = consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jetCollection"));
   fatJetToken_ =   consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("fatJetCollection"));

   edm::InputTag fixedGridRhoAllTag_ = edm::InputTag("fixedGridRhoAll", "", "RECO");   
   m_rho_token  = consumes<double>(fixedGridRhoAllTag_);

}

BESTProducer::~BESTProducer()
{

    // do anything that needs to be done at destruction time
    // (eg. close files, deallocate, resources etc.)

}
void BESTProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


   bool debug = false;
   bool debug2 = false;
   const edm::EventAuxiliary& aux = iEvent.eventAuxiliary();
   //runNum       = aux.run();
   //lumiBlockNum = aux.luminosityBlock();
   int eventNumber       = aux.event();
   if ( (jetType_ == "WB")||(jetType_ == "HT")||(jetType_ == "ZT") ) 
   {
      if (eventNumber%10 < 3)return; // want only 70% of events
   }
   if(debug)std::cout << "Starting event" << std::endl;
   if(debug)std::cout << "The jetType is " << jetType_ << std::endl;
   using namespace edm;
   using namespace fastjet;
   using namespace std;

   Handle<std::vector<pat::Jet>> fatJets;
   iEvent.getByToken(fatJetToken_, fatJets);

   Handle< std::vector<reco::GenParticle> > genPartCollection;
   iEvent.getByToken(genPartToken_, genPartCollection);

   //------------------------------------------------------------------------------
   // Gen Particles Loop ----------------------------------------------------------
   //------------------------------------------------------------------------------
   //int Suu_pdgid = 9936661;
   int chi_pdgid = 9936662;
   int nSuu = 0;
   int nChi = 0;
   int nW   = 0;
   int ntop = 0;
   int nH   = 0;
   int nSuub = 0;
   int nZ   = 0;


   std::vector<TLorentzVector> genChiZt;
   std::vector<TLorentzVector> genChiHt;
   std::vector<TLorentzVector> genChiWb;
 
   std::vector<TLorentzVector> genH;  // from Ht decay 
   std::vector<TLorentzVector> genZ;  // from Zt decay
   std::vector<TLorentzVector> genW;  // from Wb decay


   std::vector<TLorentzVector> Topb;
   std::vector<TLorentzVector> Suub;

   if(debug)std::cout << "Looking at gen particles." << std::endl;

   for (auto iG = genPartCollection->begin(); iG != genPartCollection->end(); iG++) 
   {
     if ((abs(iG->pdgId()) == 24) && ((abs(iG->mother()->pdgId()) == chi_pdgid)) ) 
     {
         genChiWb.push_back( TLorentzVector(iG->mother()->px(),iG->mother()->py(),iG->mother()->pz(),iG->mother()->energy())  );
         genW.push_back(TLorentzVector(iG->px(),iG->py(),iG->pz(),iG->energy()));
         nW++;
     }
     else if ( (abs(iG->pdgId()) == 5) && (abs(iG->mother()->pdgId()) == chi_pdgid)  )
     {
         Suub.push_back(TLorentzVector(iG->px(),iG->py(),iG->pz(),iG->energy()));
         nSuub++;
     } 

     else if ( (abs(iG->pdgId()) == 6) && ((abs(iG->mother()->pdgId()) == chi_pdgid)) ) 
     {
         ntop++;
     }
     else if ( (abs(iG->pdgId()) == 25) && ((abs(iG->mother()->pdgId()) == chi_pdgid)) ) 
     {
         genChiHt.push_back( TLorentzVector(iG->mother()->px(),iG->mother()->py(),iG->mother()->pz(),iG->mother()->energy())  );
         genH.push_back(TLorentzVector(iG->px(),iG->py(),iG->pz(),iG->energy()));
         nH++;
     }
     else if ( (abs(iG->pdgId()) == 23) && ((abs(iG->mother()->pdgId()) == chi_pdgid)) ) 
     {
         genChiZt.push_back( TLorentzVector(iG->mother()->px(),iG->mother()->py(),iG->mother()->pz(),iG->mother()->energy())  );
         genZ.push_back(TLorentzVector(iG->px(),iG->py(),iG->pz(),iG->energy()));
         nZ++;

     }
     else if ((abs(iG->pdgId()) == chi_pdgid) && (iG->isLastCopy()))
     {
         nChi++;
     }  
   }

   nEvents++;

   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////////_AK4 jets_/////////////////////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   int nAK4 = 0;
   int nTightbAK4 = 0;
   double totHT = 0;
   double dijetMassOne, dijetMassTwo;
   std::vector<TLorentzVector> leadAK4Jets;

   JME::JetResolution resolution_AK4         = JME::JetResolution::get(iSetup, "AK4PFchs_pt");              //load JER stuff from global tag
   JME::JetResolutionScaleFactor resolution_sf_AK4 = JME::JetResolutionScaleFactor::get(iSetup, "AK4PFchs");

   edm::Handle<std::vector<pat::Jet> > smallJets;
   iEvent.getByToken(jetToken_, smallJets);

   edm::Handle<double> rho;
   iEvent.getByToken(m_rho_token, rho);

   if(debug)std::cout << "Starting AK4 jet loop" << std::endl;

   for(auto iJet = smallJets->begin(); iJet != smallJets->end(); iJet++) 
   {

     double AK4_sf_total = 1.0;

     if( (iJet->pt() < 15. ) || (!(iJet->isPFJet())) || ( abs(iJet->eta()) > 2.5 )) continue;   //don't even bother with these jets, lost causes

     ////////////////////////////////////////////////////////////////////////////////////////////////////////////
     //////////   _JET ENERGY RESOLUTION STUFF //////////////////////////////////////////////////////////////////
     ////////////////////////////////////////////////////////////////////////////////////////////////////////////


      double AK4_JER_corr_factor = 1.0; // this won't be touched for data

      double sJER  = -9999.;  //JER scale factor
      double sigmaJER = -9999.;  //this is the "resolution" you get from the scale factors 
      
      //these are for getting the JER scale factors
      JME::JetParameters parameters_1;
      parameters_1.setJetPt(iJet->pt());
      parameters_1.setJetEta(iJet->eta());
      parameters_1.setRho(*rho);
      sigmaJER = resolution_AK4.getResolution(parameters_1);   //pT resolution

      //JME::JetParameters
      JME::JetParameters parameters;
      parameters.setJetPt(iJet->pt());
      parameters.setJetEta(iJet->eta());
      sJER =  resolution_sf_AK4.getScaleFactor(parameters  );  //{{JME::Binning::JetEta, iJet->eta()}});
      const reco::GenJet *genJet = iJet->genJet();
      if( genJet)   // try the first technique
      {
      AK4_JER_corr_factor = 1 + (sJER - 1)*(iJet->pt()-genJet->pt())/iJet->pt();
      }
      else   // if no gen jet is matched, try the second technique
      {
      randomNum->SetSeed( abs(static_cast<int>(iJet->phi()*1e4)) );
      double JERrand = randomNum->Gaus(0.0, sigmaJER);
      //double JERrand = 1.0 + sigmaJER;
      AK4_JER_corr_factor = max(0., 1 + JERrand*sqrt(max(pow(sJER,2)-1,0.)));   //want to make sure this is truncated at 0
      }
      AK4_sf_total*= AK4_JER_corr_factor;

     // create scaled jet object that will be used for cuts 
     pat::Jet corrJet(*iJet);
     LorentzVector corrJetP4(AK4_sf_total*iJet->px(),AK4_sf_total*iJet->py(),AK4_sf_total*iJet->pz(),AK4_sf_total*iJet->energy());
     corrJet.setP4(corrJetP4);

     //measure event HT
      if((corrJet.pt() > 30.)&&(abs(corrJet.eta()) < 2.5)  )totHT+= abs(corrJet.pt() );
     
     // apply AK4 jet selection (post JEC and JER)

      bool PUID = false;  // assumed true if not applying this
      PUID = bool( (corrJet.userInt("pileupJetIdUpdated:fullId") & (1 << 1)) || (corrJet.pt() > 50.) );

      if( (corrJet.pt()  <30.) || (!(corrJet.isPFJet())) || (!isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),PUID, corrJet.pt() )) ) continue;
      if( isHEM(corrJet.eta(), corrJet.phi())) return;

     if(nAK4 < 4)
     {
         leadAK4Jets.push_back(TLorentzVector(corrJet.px(),corrJet.py(),corrJet.pz(),corrJet.energy()));
     }
     nAK4++;

   }

   if((nAK4 <4) || (totHT < 1500.) )
   {
      
      if(debug)std::cout << "Failed nAK4,tot HT cut" << std::endl;
      return;

   } 

   if(debug)std::cout << "Passed AK4 and tot HT cuts " << std::endl;

   // calculate the candidate dijet delta R values
   double minDeltaRDisc12 = sqrt( pow(leadAK4Jets[0].DeltaR(leadAK4Jets[1]),2) + pow(leadAK4Jets[2].DeltaR(leadAK4Jets[3]),2));  // dijet one always has j1 in it
   double minDeltaRDisc13 = sqrt( pow(leadAK4Jets[0].DeltaR(leadAK4Jets[2]),2) + pow(leadAK4Jets[1].DeltaR(leadAK4Jets[3]),2));
   double minDeltaRDisc14 = sqrt( pow(leadAK4Jets[0].DeltaR(leadAK4Jets[3]),2) + pow(leadAK4Jets[1].DeltaR(leadAK4Jets[2]),2));

   
   if (  abs(min(minDeltaRDisc12, min(minDeltaRDisc13,minDeltaRDisc14)) -minDeltaRDisc12)<1e-8 ) 
   {
     //set dijet masses
     dijetMassOne = (leadAK4Jets[0] +leadAK4Jets[1]).M();
     dijetMassTwo = (leadAK4Jets[2] +leadAK4Jets[3]).M();
   }
   else if (  abs(min(minDeltaRDisc12, min(minDeltaRDisc13,minDeltaRDisc14)) -minDeltaRDisc13)<1e-8 ) 
   {
     // set dijet masses
     dijetMassOne = (leadAK4Jets[0] +leadAK4Jets[2]).M();
     dijetMassTwo = (leadAK4Jets[1] +leadAK4Jets[3]).M();
   }
   else if (  abs(min(minDeltaRDisc12, min(minDeltaRDisc13,minDeltaRDisc14)) -minDeltaRDisc14)<1e-8 ) 
   {
     //set dijet masses
     dijetMassOne = (leadAK4Jets[0] +leadAK4Jets[3]).M();
     dijetMassTwo = (leadAK4Jets[1] +leadAK4Jets[2]).M();
   }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////_AK8 Jets_/////////////////////////////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   std::vector<reco::LeafCandidate> candsBoosted;   
   std::vector<reco::LeafCandidate> candsUnboosted;

   JME::JetResolution resolution = JME::JetResolution::get(iSetup, "AK8PF_pt");              //load JER stuff from global tag
   JME::JetResolutionScaleFactor resolution_sf = JME::JetResolutionScaleFactor::get(iSetup, "AK8PF");

   //Get AK8 jet info in order to get to COM frame and get a vector of all the (good) AK8 jet particles
   int nfatjets = 0;
   int nfatjet_pre = 0;
   double tot_pt = 0;

   double tot_jet_px=0,tot_jet_py=0,tot_jet_pz=0, tot_jet_E=0;

   if(debug)std::cout << "Starting AK8 jet loop" << std::endl;

   for(auto iJet = fatJets->begin(); iJet != fatJets->end(); iJet++)  //Over AK8 Jets
   {



      if( (sqrt(pow(iJet->mass(),2)+pow(iJet->pt(),2)) < 25.) || (!(iJet->isPFJet())) || ( abs(iJet->eta()) > 2.4 )) continue;   //don't even bother with these jets

     double AK8_sf_total = 1.0;  // this scales jet/particle 4-vectors, compounds all scale factors

     ////////////////////////////////////////////////////////////////////////////////////////////////////////////
     //////////   _JET ENERGY RESOLUTION STUFF //////////////////////////////////////////////////////////////////
     ////////////////////////////////////////////////////////////////////////////////////////////////////////////
      double AK8_JER_corr_factor = 1.0; // this won't be touched for data
      

      double sJER  = -9999.;  //JER scale factor
      double sigmaJER = -9999.;  //this is the "resolution" you get from the scale factors 
      
      //these are for getting the JER scale factors
      JME::JetParameters parameters_1;
      parameters_1.setJetPt(iJet->pt());
      parameters_1.setJetEta(iJet->eta());
      parameters_1.setRho(*rho);
      sigmaJER = resolution.getResolution(parameters_1);   //pT resolution

      //JME::JetParameters
      JME::JetParameters parameters;
      parameters.setJetPt(iJet->pt());
      parameters.setJetEta(iJet->eta());
      sJER =  resolution_sf.getScaleFactor(parameters  );  //{{JME::Binning::JetEta, iJet->eta()}});

      const reco::GenJet *genJet = iJet->genJet();
      if( genJet)   // try the first technique
      {
         AK8_JER_corr_factor = 1 + (sJER - 1)*(iJet->pt()-genJet->pt())/iJet->pt();
      }
      else   // if no gen jet is matched, try the second technique
      {
         randomNum->SetSeed( abs(static_cast<int>(iJet->phi()*1e4)) );
         double JERrand = randomNum->Gaus(0.0, sigmaJER);
         //double JERrand = 1.0 + sigmaJER;
         AK8_JER_corr_factor = max(0., 1 + JERrand*sqrt(max(pow(sJER,2)-1,0.)));   //want to make sure this is truncated at 0
      }
      AK8_sf_total*= AK8_JER_corr_factor;
     
     /////////////////////////////////////////////////////////////////////////////////////////////////////////////
     /////////////////////////////////////////////////////////////////////////////////////////////////////////////

     // create scaled jet object that will be used for cuts 
     pat::Jet corrJet(*iJet);
     LorentzVector corrJetP4(AK8_sf_total*iJet->px(),AK8_sf_total*iJet->py(),AK8_sf_total*iJet->pz(),AK8_sf_total*iJet->energy());
     corrJet.setP4(corrJetP4);

     tot_pt+= corrJet.pt();

      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 45.)) 
      {
         nfatjet_pre++;
      }
      if((sqrt(pow(corrJet.mass(),2)+pow(corrJet.pt(),2)) < 200.) || (!(corrJet.isPFJet())) || (!isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets )) || (corrJet.mass()< 0.)) continue; //userFloat("ak8PFJetsPuppiSoftDropMass")
      //if(isHEM(corrJet.eta(),corrJet.phi()))continue;

     for (unsigned int iii=0; iii<iJet->numberOfDaughters();iii++)   // get all jet particles
     {
      const reco::Candidate* iJ = iJet->daughter(iii);
      const pat::PackedCandidate* candJetbegin = (pat::PackedCandidate*) iJ;
      double puppiweight = candJetbegin->puppiWeight();
      candsUnboosted.push_back(reco::LeafCandidate(iJet->daughter(iii)->charge(), Particle::LorentzVector(AK8_sf_total*puppiweight*iJ->px(), AK8_sf_total*puppiweight*iJ->py(), AK8_sf_total*puppiweight*iJ->pz(), AK8_sf_total*puppiweight*iJ->energy())));
      tot_jet_px+=AK8_sf_total*puppiweight*candJetbegin->px();tot_jet_py+=AK8_sf_total*puppiweight*candJetbegin->py();tot_jet_pz+=AK8_sf_total*puppiweight*candJetbegin->pz();tot_jet_E+=AK8_sf_total*puppiweight*candJetbegin->energy();
     }
     nfatjets++;
   }



   //if (jetType_ == "QCD" || jetType_ == "Top")  // VLQs have very high efficiency rates here,  but a small portion of stats are lost, so this won't be applied to them
   //{
   if (  (nfatjets < 3) ||   ((nfatjet_pre < 2) && ((dijetMassOne < 1000.) || (dijetMassTwo < 1000.)  )  )  ) 
   {
      if(debug)std::cout << "Failed nfatjet, nfatjet_pre, or dijet cut" << std::endl;
      return;
   }  
   //}



   if(debug2)std::cout << "Passed AK8 and AK4 dijet cuts " << std::endl;

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////  _clustering  ///////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



   if(debug)std::cout << "Doing jet clustering." << std::endl;

   double tot_jet_p = sqrt(pow(tot_jet_px,2)+pow(tot_jet_py,2)+pow(tot_jet_pz,2)   ); 

   TVector3 totJetBeta = TVector3(tot_jet_px/tot_jet_E,tot_jet_py/tot_jet_E ,tot_jet_pz/tot_jet_E);


   std::vector<TLorentzVector> genChiZtBoosted;
   std::vector<TLorentzVector> genChiHtBoosted;
   std::vector<TLorentzVector> genChiWbBoosted;
   //boost gen Chis into MPP frame
   if(debug)std::cout << "Boosting gen chis." << std::endl;

   if ( genChiZt.size() > 0)
   {
     for(auto iG = genChiZt.begin(); iG != genChiZt.end(); iG++ )
     {
      iG->Boost(-totJetBeta.X(),-totJetBeta.Y(),-totJetBeta.Z());
      genChiZtBoosted.push_back(TLorentzVector(iG->Px(),iG->Py(),iG->Pz(),iG->E()) );
     }   
   }
   if ( genChiHt.size() > 0)
   {
     for(auto iG = genChiHt.begin(); iG != genChiHt.end(); iG++ )
     {
      iG->Boost(-totJetBeta.X(),-totJetBeta.Y(),-totJetBeta.Z());
      genChiHtBoosted.push_back(TLorentzVector(iG->Px(),iG->Py(),iG->Pz(),iG->E()) );
     }   
   }
   if ( genChiWb.size() > 0)
   {
     for(auto iG = genChiWb.begin(); iG != genChiWb.end(); iG++ )
     {
      iG->Boost(-totJetBeta.X(),-totJetBeta.Y(),-totJetBeta.Z());
      genChiWbBoosted.push_back(TLorentzVector(iG->Px(),iG->Py(),iG->Pz(),iG->E()) );
     }   
   }
   if ( ( genChiHt.size() + genChiZt.size() + genChiWb.size() ) > 2)
   {
     std::cout << "More than 2 gen Chis ... you did something wrong " << std::endl;
   }

   if(debug)std::cout << "Boosting all jet particles to COM frame." << std::endl;


   //////////boost all jet particles into MPP frame
   std::vector<fastjet::PseudoJet> candsBoostedPJ;

   for(auto iC = candsUnboosted.begin();iC != candsUnboosted.end(); iC++)
   {
     TLorentzVector iC_(iC->px(),iC->py(),iC->pz(),iC->energy());
     iC_.Boost(-totJetBeta.X(),-totJetBeta.Y(),-totJetBeta.Z());
     candsBoostedPJ.push_back(  fastjet::PseudoJet(iC_.Px(), iC_.Py(), iC_.Pz(), iC_.E()   ) );

   }


   if(debug)std::cout << "Doing R=0.8 jet clustering." << std::endl;

   double R0 = 0.8;
   //fastjet::JetDefinition jet_def0(fastjet::antikt_algorithm, R0);
   fastjet::JetDefinition jet_def0(fastjet::cambridge_algorithm, R0);
   fastjet::ClusterSequence cs_jet0(candsBoostedPJ, jet_def0); 
   //std::vector<TLorentzVector> candsBoostedTLV;   



   // shedding particles here -> can change this energy threshold, just changed it from 50. 
   std::vector<fastjet::PseudoJet> jetsFJ_jet0 = fastjet::sorted_by_E(cs_jet0.inclusive_jets(10.));
   if(debug)std::cout << "Collecting boosted, reclustered jet particles to calculate thrust." << std::endl;


   // I suspect it's getting stuck here ...
  
   for (auto iJet=jetsFJ_jet0.begin(); iJet<jetsFJ_jet0.end(); iJet++)               
   {   
     
     if (iJet->constituents().size() > 0)
     {
         int counter = 0;
         for(auto iDaughter = iJet->constituents().begin(); iDaughter != iJet->constituents().end(); iDaughter++)
         {

            candsBoosted.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iDaughter->px(), iDaughter->py(), iDaughter->pz(), iDaughter->E())));
            //candsBoostedTLV.push_back(TLorentzVector( iDaughter->px(), iDaughter->py(), iDaughter->pz(), iDaughter->E() ));
            counter++;
            if(counter > 200)
            {
               std::cout << "Bug in pseudojet loop, exiting ..." << std::endl;
               return;
            } // avoiding bug here involving pushing back particles
         }
     }
   }

   if(debug)std::cout << "Calculating thrust." << std::endl;

   Thrust thrust_(candsBoosted.begin(), candsBoosted.end());               //thrust axis in COM frame
   math::XYZVector thrustAxis = thrust_.axis();
   TVector3 thrust_vector(thrustAxis.X(),thrustAxis.Y(),thrustAxis.Z());




   ////////////////// determine superjet classifications ///////////////////////
   bool handSort = false;
   std::string positiveLobe = "";
   std::string negativeLobe = "";
   if ( (jetType_ == "WB") || (jetType_ == "HT") || (jetType_ == "ZT"))
   {
      if(debug)std::cout << "Characterizing superjets by matching to gen chis." << std::endl;

      //positiveLobe = positive superjet classification: 1 = Zt, 2 = Ht, 3 = Wb


      TLorentzVector chi1(0,0,0,0);

      if ( genChiWbBoosted.size() > 0)
      {
        for(auto iG = genChiWbBoosted.begin(); iG != genChiWbBoosted.end(); iG++ )
        {
         // need angle between thrust vector
         TVector3 candJet_vec = iG->Vect();

         if( abs(chi1.Px())< 1e-8)
         {
            chi1.SetPxPyPzE(iG->Px(), iG->Py(),iG->Pz(),iG->E());
         }

         double cosAngle = cos(candJet_vec.Angle(thrust_vector));
         if (cosAngle > 0)
         {
            if ( positiveLobe == "") 
            {
              positiveLobe = "WB";
            }
            else 
            {
              std::cout << "Two Chis on one side of SJ ... problem. " << std::endl;

               // clear the gen vectors here, most likely what is happening is the thrust axis is perpendicular to the SJ axis 
               // assign superjets in a more sophisticated manner ... 
              //std::cout << "The angle between them is " << candJet_vec.Angle(chi1.Vect()) << std::endl;
              handSort = true;
            }
         }
         else if (cosAngle < 0)
         {
            if ( negativeLobe == "") 
            {
              negativeLobe = "WB";
            }
            else 
            {
              std::cout << "Two Chis on one side of SJ ... problem. " << std::endl;
              //std::cout << "The angle between them is " << candJet_vec.Angle(chi1.Vect());

              handSort = true;
            }
         }
        }   
      }

   //need to change these ones.
      if ( genChiZtBoosted.size() > 0)
      {


        for(auto iG = genChiZtBoosted.begin(); iG != genChiZtBoosted.end(); iG++ )
        {

         if( abs(chi1.Px())< 1e-8)
         {
            chi1.SetPxPyPzE(iG->Px(), iG->Py(),iG->Pz(),iG->E());
         }
         // need angle between thrust vector
         TVector3 candJet_vec = iG->Vect();
         double cosAngle = cos(candJet_vec.Angle(thrust_vector));
         if (cosAngle > 0)
         {
            if ( positiveLobe == "") 
            {
              positiveLobe = "ZT";
            }
            else 
            {
              std::cout << "Two Chis on one side of SJ ... problem. " << std::endl;
              //std::cout << "The angle between them is " << candJet_vec.Angle(chi1.Vect());

              handSort = true;
            }
         }
         else if (cosAngle < 0)
         {
            if ( negativeLobe == "")
            {
              negativeLobe = "ZT";
            } 
            else 
            {
              std::cout << "Two Chis on one side of SJ ... problem. " << std::endl;
              //std::cout << "The angle between them is " << candJet_vec.Angle(chi1.Vect());

              handSort = true;
            }
         }
        }   
      }
      if ( genChiHtBoosted.size() > 0)
      {

        for(auto iG = genChiHtBoosted.begin(); iG != genChiHtBoosted.end(); iG++ )
        {
         if( abs(chi1.Px())< 1e-8)
         {
            chi1.SetPxPyPzE(iG->Px(), iG->Py(),iG->Pz(),iG->E());
         }
         // need angle between thrust vector
         TVector3 candJet_vec = iG->Vect();
         double cosAngle = cos(candJet_vec.Angle(thrust_vector));
         if (cosAngle > 0)
         {
            if ( positiveLobe == "")
            {
              positiveLobe = "HT";
            } 
            else 
            {
              std::cout << "Two Chis on one side of SJ ... problem. " << std::endl;
              //std::cout << "The angle between them is " << candJet_vec.Angle(chi1.Vect());
              handSort = true;
            }
         }
         else if (cosAngle < 0)
         {
            if ( negativeLobe == "")
            {
              negativeLobe = "HT";
            } 
            else 
            {
              std::cout << "Two Chis on one side of SJ ... problem. " << std::endl;
              //std::cout << "The angle between them is " << candJet_vec.Angle(chi1.Vect());
              handSort = true;
            }
         }
        }   
      }


      if(debug) std::cout << "The event has " <<genChiHtBoosted.size() << "/" << genChiWbBoosted.size() << "/"<< genChiZtBoosted.size() << " matched HT/WB/ZT superjets." << std::endl;

   }


   //sort jets in terms of angle relative to thrust axis

   std::vector<TLorentzVector> negSuperJet_preSort;
   std::vector<TLorentzVector> posSuperJet_preSort;
   std::vector<TLorentzVector> miscJets;
   for (auto iJet=jetsFJ_jet0.begin(); iJet<jetsFJ_jet0.end(); iJet++)                             
   {
     TLorentzVector candJet(iJet->px(),iJet->py(),iJet->pz(),iJet->E());
     TVector3 candJet_vec = candJet.Vect();
     double cosAngle = cos(candJet_vec.Angle(thrust_vector));
     if    (cosAngle < -0.85) negSuperJet_preSort.push_back(candJet);  
     else if(cosAngle > 0.85) posSuperJet_preSort.push_back(candJet);
     else{ miscJets.push_back(candJet); }
   }

   if(debug)std::cout << "There are "<<  posSuperJet_preSort.size() << "/" <<negSuperJet_preSort.size()  << "/" << miscJets.size()<< "positive/negative/misc pre-sort AK8 jets per superjet." << std::endl;
   std::vector<TLorentzVector> negSuperJet;
   std::vector<TLorentzVector> posSuperJet;

   if(( miscJets.size() > 0) )
   {   
      sortJets testing(negSuperJet_preSort,posSuperJet_preSort,miscJets);
      negSuperJet = testing.finalSuperJet1;
      posSuperJet = testing.finalSuperJet2;
   }
   else
   {
      negSuperJet = negSuperJet_preSort;
      posSuperJet = posSuperJet_preSort;
   }

   if(debug)std::cout << "There are "<< posSuperJet.size() << "/" << negSuperJet.size() << "positive/negative post sort AK8 jets per superjet." << std::endl;



   // hand sort the superjets if there was a problem earlier in assignment 
   if(handSort)
   {
      negativeLobe = "";
      positiveLobe = "";
      // find the main VLQ daughter (H,T,W,Z) of the jetType_ and assign it to the SJ that has the closest reclustered associated AK8 jet in delta R
      // this is not a great way to do this, but more accurate than nothing
      
      if(jetType_ == "HT")
      {
         // loop over the gen Hs 
         for(auto iH = genH.begin(); iH!=genH.end();iH++)
         {
            double minAngle = 9e15;
            for(auto iJet = posSuperJet.begin(); iJet != posSuperJet.end(); iJet++)
            {
               TVector3 iH_vec = iH->Vect();
               double angle = abs(iH_vec.Angle(iJet->Vect()));
               if(angle < minAngle)
               {
                  minAngle     = angle;
                  positiveLobe = "HT";
               }
            }
            for(auto iJet = negSuperJet.begin(); iJet != negSuperJet.end(); iJet++)
            {
               TVector3 iH_vec = iH->Vect();
               double angle = abs(iH_vec.Angle(iJet->Vect()));
               if(angle < minAngle)
               {
                  minAngle     = angle;
                  negativeLobe = "HT";
               }
            }
         }
      }
      else if(jetType_ == "WB")
      {
         for(auto iW = genW.begin(); iW!=genW.end();iW++)
         {
            double minAngle = 9e15;
            for(auto iJet = posSuperJet.begin(); iJet != posSuperJet.end(); iJet++)
            {
               TVector3 iW_vec = iW->Vect();
               double angle = abs(iW_vec.Angle(iJet->Vect()));
               if(angle < minAngle)
               {
                  minAngle     = angle;
                  positiveLobe = "WB";
               }
            }
            for(auto iJet = negSuperJet.begin(); iJet != negSuperJet.end(); iJet++)
            {
               TVector3 iW_vec = iW->Vect();
               double angle = abs(iW_vec.Angle(iJet->Vect()));
               if(angle < minAngle)
               {
                  minAngle     = angle;
                  negativeLobe = "WB";
               }   
            }
         }
      }
      else if(jetType_ == "ZT")
      {
         for(auto iZ = genZ.begin(); iZ!=genZ.end();iZ++)
         {
            double minAngle = 9e15;
            for(auto iJet = posSuperJet.begin(); iJet != posSuperJet.end(); iJet++)
            {
               TVector3 iZ_vec = iZ->Vect();
               double angle = abs(iZ_vec.Angle(iJet->Vect()));
               if(angle < minAngle)
               {
                  minAngle     = angle;
                  positiveLobe = "ZT";
               }   
            }
            for(auto iJet = negSuperJet.begin(); iJet != negSuperJet.end(); iJet++)
            {
               TVector3 iZ_vec = iZ->Vect();
               double angle = abs(iZ_vec.Angle(iJet->Vect()));
               if(angle < minAngle)
               {
                  minAngle     = angle;
                  negativeLobe = "ZT";
               }   
            }
         }
      }     
      if ((jetType_ != positiveLobe) && (jetType_ != negativeLobe))
      {
         return; // nothing to do here, things still didn't work
      } 
      else
      {
         std::cout << "double chi problem resolved. " << std::endl;
      }
   }



   std::vector<fastjet::PseudoJet> superJetOne;     //jets for dot prduct #cos(theta) > 0
   std::vector<fastjet::PseudoJet> superJetTwo;      //jets for dot product #cos(theta) < 0 
   std::vector<TLorentzVector> superJetOneLV;   //workaround for weird fastjet bug
   std::vector<TLorentzVector> superJetTwoLV;   //workaround for weird fastjet bug


   if( (negSuperJet.size() < 1 ) || (posSuperJet.size()<1)   ) 
   {
     std::cout << "One superjet vector has size 0 - " << negSuperJet.size()<< " / " << posSuperJet.size()<< std::endl; 
     return;
   }

   TLorentzVector SJ1(0,0,0,0);
   TLorentzVector SJ2(0,0,0,0);

   for(unsigned int iii = 0; iii < negSuperJet.size();iii++)
   {
     SJ2+=negSuperJet[iii];
   }

   for(unsigned int iii = 0; iii < posSuperJet.size();iii++)
   {
     SJ1+=posSuperJet[iii];
   }

 /// don't need this any more with new technique
   //get vector of pseudo jet particles for all particles in each superjet, one per superjet
   for (auto iJet=jetsFJ_jet0.begin(); iJet<jetsFJ_jet0.end(); iJet++)              
   {   
     TLorentzVector candJet(iJet->px(),iJet->py(),iJet->pz(),iJet->E());

     int SJMatch = 0;
     if    (isMatchedtoSJ(posSuperJet,candJet)) SJMatch = 1;
     else if (isMatchedtoSJ(negSuperJet,candJet)) SJMatch = 2;
     else if(SJMatch == 0) std::cout << "Something didn't work... " << std::endl;

     //sort jet particles into either SuperJet 1 or SuperJet 2
     if (iJet->constituents().size() > 0)
     {
         for(auto iJ = iJet->constituents().begin(); iJ != iJet->constituents().end(); iJ++)
         {  
            if    (SJMatch==1)superJetOneLV.push_back(TLorentzVector(iJ->px(),iJ->py(),iJ->pz(),iJ->E()));
            else if(SJMatch==2)superJetTwoLV.push_back(TLorentzVector(iJ->px(),iJ->py(),iJ->pz(),iJ->E()));
         }
      }   
   }

   for(auto iPart = superJetOneLV.begin(); iPart != superJetOneLV.end(); iPart++)
   {
      superJetOne.push_back(fastjet::PseudoJet(iPart->Px(),iPart->Py(),iPart->Pz(),iPart->E()));
   }
   for(auto iPart = superJetTwoLV.begin(); iPart != superJetTwoLV.end(); iPart++)
   {
     superJetTwo.push_back(fastjet::PseudoJet(iPart->Px(),iPart->Py(),iPart->Pz(),iPart->E()));
   }

   if(debug)std::cout << "Superjet particles have been collected and boosted." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////// superjet analysis /////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   double superJetpx,superJetpy,superJetpz,superJetE;
   std::vector<std::vector<fastjet::PseudoJet>> superJets;
   superJets.push_back(superJetOne);
   superJets.push_back(superJetTwo);

   //superjet classification: 1 = Zt, 2 = Ht, 3 = Wb
   //these look fine -> move this into superjet COM frame and compare vs the different AK4 jets
   //maybe try looking at ALL particles in SJ COM frame, not just the leading 4 AK4 jets



   //////////////////////////////////////////////////////////////////////
   ///////////////////////// _superjet loop /////////////////////////////
   //////////////////////////////////////////////////////////////////////

   std::vector<fastjet::PseudoJet> SJParts1;   
   std::vector<fastjet::PseudoJet> SJParts2; 

   // superjet 1 will be first in loop -> SJ1 
   // pos SJ means positive cosine relative to thrust axis 
   // superjet 1 = positive superjet 

   int nSuperJets = 0;
   int SJCounter = 0;
   if(debug)std::cout << "Looking at superjets " << std::endl;

   //std::cout << "SJ1 has " << superJetOne.size() << " particles, and SJ2 has " << superJetTwo.size() << " particles" << std::endl;
   for(auto iSJ = superJets.begin();iSJ!= superJets.end();iSJ++)
   {

     SJCounter++;
      if(debug)std::cout << "Looking at superjet " << SJCounter << " -------------------- " << std::endl;

     //if    (  ((SJCounter == 1) && (positiveLobe != jetType_)) || ((SJCounter == 2) && (negativeLobe != jetType_)) ) return;
     if( (jetType_ == "HT") || (jetType_ == "WB") || (jetType_ == "ZT")    ) // only want the correct superjets saved for signal training
     {
         if((SJCounter == 1) && (positiveLobe != jetType_)) continue;
         else if((SJCounter == 2) && (negativeLobe != jetType_)) continue;
     }
     

     //std::cout << "In SJ loop for jet type " << jetType_ << std::endl;
     // bases covered here 


     // loop over individual superjet particles 
     superJetpx=0;superJetpy=0;superJetpz=0;superJetE=0;
     for(auto iP = iSJ->begin();iP != iSJ->end();iP++)
     {
         superJetpx+=iP->px();superJetpy+=iP->py();superJetpz+=iP->pz();superJetE +=iP->E();
     }

     TLorentzVector superJetTLV(superJetpx,superJetpy,superJetpz,superJetE);  //Lorentz vector representing jet axis -> now minimize the parallel momentum
     
     //boost COM
     std::vector<fastjet::PseudoJet> boostedSuperJetPart;

     //boost particles in SuperJet to COM frame
     for(auto iP = iSJ->begin();iP != iSJ->end();iP++)
     {
      TLorentzVector iP_(iP->px(),iP->py(),iP->pz(),iP->E());
      iP_.Boost(-superJetTLV.Px()/superJetTLV.E(),-superJetTLV.Py()/superJetTLV.E(),-superJetTLV.Pz()/superJetTLV.E());
      boostedSuperJetPart.push_back(fastjet::PseudoJet(iP_.Px(),iP_.Py(),iP_.Pz(),iP_.E()));
     }

     ///////////get a bunch of versions of SJ particle collection to calclate BES variables
     std::vector<TLorentzVector> boostedSuperJetPart_TLV;
     std::vector<reco::LeafCandidate> boostedSuperJetPart_LC;
     std::vector<math::XYZVector> boostedSuperJetPart_XYZ;
     double sumPz = 0, sumP = 0;
     for(auto iP_= boostedSuperJetPart.begin(); iP_ !=boostedSuperJetPart.end(); iP_++)
     {   
      boostedSuperJetPart_TLV.push_back(TLorentzVector(iP_->px(),iP_->py(),iP_->pz(),iP_->E()));
      boostedSuperJetPart_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP_->px(),iP_->py(),iP_->pz(),iP_->E())));
      boostedSuperJetPart_XYZ.push_back(math::XYZVector( iP_->px(),iP_->py(),iP_->pz() ));
      sumPz += abs(iP_->pz());
      sumP += abs(sqrt(pow(iP_->pz(),2) + pow(iP_->px(),2)+ pow(iP_->py(),2)));
     }


     ///reclustering SuperJet that is now boosted into the SJ COM frame
     double R = 0.4;


     if(debug)std::cout << "Clustering boosted superjet particles into new CA4 jets." << std::endl;

     //fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
     fastjet::JetDefinition jet_def(fastjet::cambridge_algorithm, R);
     fastjet::ClusterSequence cs_jet(boostedSuperJetPart, jet_def); 
     std::vector<fastjet::PseudoJet> jetsFJ_jet = fastjet::sorted_by_E(cs_jet.inclusive_jets(0.0));


     double SJ_25_px = 0, SJ_25_py=0,SJ_25_pz=0,SJ_25_E=0;
     double SJ_50_px = 0, SJ_50_py=0,SJ_50_pz=0,SJ_50_E=0;
     double SJ_75_px = 0, SJ_75_py=0,SJ_75_pz=0,SJ_75_E=0;
     double SJ_100_px = 0, SJ_100_py=0,SJ_100_pz=0,SJ_100_E=0;
     double SJ_150_px = 0, SJ_150_py=0,SJ_150_pz=0,SJ_150_E=0;
     double SJ_200_px = 0, SJ_200_py=0,SJ_200_pz=0,SJ_200_E=0;
     double SJ_300_px = 0, SJ_300_py=0,SJ_300_pz=0,SJ_300_E=0;
     double SJ_400_px = 0, SJ_400_py=0,SJ_400_pz=0,SJ_400_E=0;
     double SJ_500_px = 0, SJ_500_py=0,SJ_500_pz=0,SJ_500_E=0;
     double SJ_800_px = 0, SJ_800_py=0,SJ_800_pz=0,SJ_800_E=0;
     double SJ_1000_px = 0, SJ_1000_py=0,SJ_1000_pz=0,SJ_1000_E=0;

     int SJ_nAK4_1 = 0, SJ_nAK4_5 = 0, SJ_nAK4_10_ = 0;
     int SJ_nAK4_25_ = 0;
     int SJ_nAK4_50_ = 0, SJ_nAK4_75_ = 0, SJ_nAK4_100_ = 0,SJ_nAK4_150_ = 0,SJ_nAK4_200_ = 0,SJ_nAK4_300_ = 0;
     int SJ_nAK4_400_ = 0,SJ_nAK4_500_ = 0,SJ_nAK4_800_ = 0,SJ_nAK4_1000_ = 0;


     // particles in each superjet in the SJ COM frame
     std::vector<TLorentzVector> AK41_parts;
     std::vector<TLorentzVector> AK42_parts;
     std::vector<TLorentzVector> AK43_parts;
     std::vector<TLorentzVector> AK44_parts;

     double AK41_px = 0, AK41_py=0,AK41_pz = 0, AK41_E = 0;
     double AK42_px = 0, AK42_py=0,AK42_pz = 0, AK42_E = 0;
     double AK43_px = 0, AK43_py=0,AK43_pz = 0, AK43_E = 0;
     double AK44_px = 0, AK44_py=0,AK44_pz = 0, AK44_E = 0;

     if (jetsFJ_jet.size() < 4)
     {
      std::cout << "A pseudojet vector has a size smaller than 4 - not reclustering many jets from whole pool of particles - " << jetsFJ_jet.size() << std::endl;
      return;
     }
     int pseudoJetNum = 0;
     for (auto iPJ=jetsFJ_jet.begin(); iPJ<jetsFJ_jet.end(); iPJ++)               
     {

         // do calculations of AK4 btagged particle ratios for leading 4 AK4 jets
         std::vector<fastjet::PseudoJet> iPJ_daughters = iPJ->constituents();

         if ( pseudoJetNum < 4)
         {
            for(auto iPart = iPJ_daughters.begin(); iPart != iPJ_daughters.end(); iPart++)
            {
              if(nSuperJets == 0)SJParts1.push_back( fastjet::PseudoJet( iPart->px(), iPart->py(),iPart->pz(),iPart->E()   )   );
              else if(nSuperJets == 0)SJParts2.push_back( fastjet::PseudoJet( iPart->px(), iPart->py(),iPart->pz(),iPart->E() )   );

            }
         }

         if(debug)std::cout << "Looking at reclustered superjet CA4 jets." << std::endl;

         if(pseudoJetNum == 0)
         {   
            for(auto iPart = iPJ_daughters.begin(); iPart != iPJ_daughters.end(); iPart++)
            {

              AK41_parts.push_back(TLorentzVector(iPart->px(),iPart->py(),iPart->pz(),iPart->E()));
              AK41_px+=iPart->px();AK41_py+=iPart->py();AK41_pz+=iPart->pz();AK41_E+=iPart->E();
            }
         }
         else if(pseudoJetNum == 1)
         {
            for(auto iPart = iPJ_daughters.begin(); iPart != iPJ_daughters.end(); iPart++)
            {
              AK42_parts.push_back(TLorentzVector(iPart->px(),iPart->py(),iPart->pz(),iPart->E()));
              AK42_px+=iPart->px();AK42_py+=iPart->py();AK42_pz+=iPart->pz();AK42_E+=iPart->E();
            }
         }
         else if(pseudoJetNum == 2)
         {
            for(auto iPart = iPJ_daughters.begin(); iPart != iPJ_daughters.end(); iPart++)
            {
              AK43_parts.push_back(TLorentzVector(iPart->px(),iPart->py(),iPart->pz(),iPart->E()));
              AK43_px+=iPart->px();AK43_py+=iPart->py();AK43_pz+=iPart->pz();AK43_E+=iPart->E();
            }
         }
         else if(pseudoJetNum == 3)
         {
            for(auto iPart = iPJ_daughters.begin(); iPart != iPJ_daughters.end(); iPart++)
            {
              AK44_parts.push_back(TLorentzVector(iPart->px(),iPart->py(),iPart->pz(),iPart->E()));
              AK44_px+=iPart->px();AK44_py+=iPart->py();AK44_pz+=iPart->pz();AK44_E+=iPart->E();

            }
         }
         if(debug)std::cout << "Looking at reclustered CA4 jets with energy thresholds." << std::endl;


         if(iPJ->E()>1.)
         {
            SJ_nAK4_1++;   // this is just used for a fraction calculation
         }
         if(iPJ->E()>5.)
         {
            SJ_nAK4_5++;   // this is just used for a fraction calculation
         }
         if(iPJ->E()>10.)
         {
            SJ_nAK4_10_++;
         }
         if(iPJ->E()>25.)
         {
            SJ_25_px+=iPJ->px();SJ_25_py+=iPJ->py();SJ_25_pz+=iPJ->pz();SJ_25_E+=iPJ->E();
            SJ_nAK4_25_++;
         }

         if(iPJ->E()>50.)
         {
            SJ_50_px+=iPJ->px();SJ_50_py+=iPJ->py();SJ_50_pz+=iPJ->pz();SJ_50_E+=iPJ->E();
            SJ_nAK4_50_++;
         }
         if(iPJ->E()>75.)
         {
            SJ_75_px+=iPJ->px();SJ_75_py+=iPJ->py();SJ_75_pz+=iPJ->pz();SJ_75_E+=iPJ->E();
            SJ_nAK4_75_++;
         }
         if(iPJ->E()>100)
         {
            SJ_100_px+=iPJ->px();SJ_100_py+=iPJ->py();SJ_100_pz+=iPJ->pz();SJ_100_E+=iPJ->E();
            SJ_nAK4_100_++; 
         }

         if(iPJ->E()>150)
         {
            SJ_150_px+=iPJ->px();SJ_150_py+=iPJ->py();SJ_150_pz+=iPJ->pz();SJ_150_E+=iPJ->E();
            SJ_nAK4_150_++; 
         }

         if(iPJ->E()>200)
         {
            SJ_200_px+=iPJ->px();SJ_200_py+=iPJ->py();SJ_200_pz+=iPJ->pz();SJ_200_E+=iPJ->E();
            SJ_nAK4_200_++; 
         }
         if(iPJ->E()>300)
         {
            SJ_300_px+=iPJ->px();SJ_300_py+=iPJ->py();SJ_300_pz+=iPJ->pz();SJ_300_E+=iPJ->E();
            SJ_nAK4_300_++; 
         }
         if(iPJ->E()>400)
         {
            SJ_400_px+=iPJ->px();SJ_400_py+=iPJ->py();SJ_400_pz+=iPJ->pz();SJ_400_E+=iPJ->E();
            SJ_nAK4_400_++; 
         }
         if(iPJ->E()>500)
         {
            SJ_500_px+=iPJ->px();SJ_500_py+=iPJ->py();SJ_500_pz+=iPJ->pz();SJ_500_E+=iPJ->E();
            SJ_nAK4_500_++; 
         }
         if(iPJ->E()>800)
         {
            SJ_800_px+=iPJ->px();SJ_800_py+=iPJ->py();SJ_800_pz+=iPJ->pz();SJ_800_E+=iPJ->E();
            SJ_nAK4_800_++; 
         }
         if(iPJ->E()>1000)
         {
            SJ_1000_px+=iPJ->px();SJ_1000_py+=iPJ->py();SJ_1000_pz+=iPJ->pz();SJ_1000_E+=iPJ->E();
            SJ_nAK4_1000_++; 
         }

         pseudoJetNum++;
     }
     boostedSuperJetPart.clear();   //shouldn't be needed, just in case

     nSuperJets++; 


     /////annoying process of getting the BES information for the reclustered AK4 jets

     if(debug)std::cout << "Calculating BES variables." << std::endl;

     TVector3 AK41_boost(AK41_px/AK41_E,AK41_py/AK41_E,AK41_pz/AK41_E);
     TVector3 AK42_boost(AK42_px/AK42_E,AK42_py/AK42_E,AK42_pz/AK42_E);
     TVector3 AK43_boost(AK43_px/AK43_E,AK43_py/AK43_E,AK43_pz/AK43_E);
     TVector3 AK44_boost(AK44_px/AK44_E,AK44_py/AK44_E,AK44_pz/AK44_E);

     std::vector<TLorentzVector> boostedAK41_Part_TLV;
     std::vector<reco::LeafCandidate>  boostedAK41_Part_LC;
     std::vector<math::XYZVector>  boostedAK41_Part_XYZ;

     std::vector<TLorentzVector> boostedAK42_Part_TLV;
     std::vector<reco::LeafCandidate>  boostedAK42_Part_LC;
     std::vector<math::XYZVector>  boostedAK42_Part_XYZ;

     std::vector<TLorentzVector> boostedAK43_Part_TLV;
     std::vector<reco::LeafCandidate>  boostedAK43_Part_LC;
     std::vector<math::XYZVector>  boostedAK43_Part_XYZ;

     std::vector<TLorentzVector> boostedAK44_Part_TLV;
     std::vector<reco::LeafCandidate>  boostedAK44_Part_LC;
     std::vector<math::XYZVector>  boostedAK44_Part_XYZ;

     double sumPz_AK41 =0,sumPz_AK42 = 0,sumPz_AK43 = 0,sumPz_AK44 = 0;
     double sumP_AK41 = 0, sumP_AK42 = 0,sumP_AK43 = 0, sumP_AK44 = 0;

     int n_AK41_parts_1 = 0, n_AK41_parts_5 = 0, n_AK41_parts_10 =0; // n_AK41_parts_20=0, n_AK41_parts_40=0, n_AK41_parts_50=0, n_AK41_parts_75=0, n_AK41_parts_100=0;
     int n_AK41_parts_0p1 = 0, n_AK41_parts_0p5 = 0, n_AK41_parts_2 =0, n_AK41_parts_7p5=0, n_AK41_parts_15=0;
     int n_AK42_parts_1 = 0, n_AK42_parts_5 = 0, n_AK42_parts_10 =0; 
     int n_AK42_parts_0p1 = 0, n_AK42_parts_0p5 = 0, n_AK42_parts_2 =0, n_AK42_parts_7p5=0, n_AK42_parts_15=0;
     int n_AK43_parts_1 = 0, n_AK43_parts_5 = 0, n_AK43_parts_10 =0; 
     int n_AK43_parts_0p1 = 0, n_AK43_parts_0p5 = 0, n_AK43_parts_2 =0, n_AK43_parts_7p5=0, n_AK43_parts_15=0;
     int n_AK44_parts_1 = 0, n_AK44_parts_5 = 0, n_AK44_parts_10 =0; 
     int n_AK44_parts_0p1 = 0, n_AK44_parts_0p5 = 0, n_AK44_parts_2 =0, n_AK44_parts_7p5=0, n_AK44_parts_15=0;


     TLorentzVector AK41_0p1(0,0,0,0); TLorentzVector AK41_0p5(0,0,0,0); TLorentzVector AK41_1(0,0,0,0); 
     TLorentzVector AK41_2(0,0,0,0); TLorentzVector AK41_5(0,0,0,0); TLorentzVector AK41_7p5(0,0,0,0);
     TLorentzVector AK41_10(0,0,0,0); TLorentzVector AK41_15(0,0,0,0);
     TLorentzVector AK41_20(0,0,0,0); //TLorentzVector AK41_40(0,0,0,0);
     //TLorentzVector AK41_50(0,0,0,0); TLorentzVector AK41_75(0,0,0,0); TLorentzVector AK41_100(0,0,0,0);
     TLorentzVector AK42_0p1(0,0,0,0); TLorentzVector AK42_0p5(0,0,0,0); TLorentzVector AK42_1(0,0,0,0); 
     TLorentzVector AK42_2(0,0,0,0); TLorentzVector AK42_5(0,0,0,0); TLorentzVector AK42_7p5(0,0,0,0);
     TLorentzVector AK42_10(0,0,0,0); TLorentzVector AK42_15(0,0,0,0);
     TLorentzVector AK42_20(0,0,0,0); 
     TLorentzVector AK43_0p1(0,0,0,0); TLorentzVector AK43_0p5(0,0,0,0); TLorentzVector AK43_1(0,0,0,0); 
     TLorentzVector AK43_2(0,0,0,0); TLorentzVector AK43_5(0,0,0,0); TLorentzVector AK43_7p5(0,0,0,0);
     TLorentzVector AK43_10(0,0,0,0); TLorentzVector AK43_15(0,0,0,0);
     TLorentzVector AK43_20(0,0,0,0); 
     TLorentzVector AK44_0p1(0,0,0,0); TLorentzVector AK44_0p5(0,0,0,0); TLorentzVector AK44_1(0,0,0,0); 
     TLorentzVector AK44_2(0,0,0,0); TLorentzVector AK44_5(0,0,0,0); TLorentzVector AK44_7p5(0,0,0,0);
     TLorentzVector AK44_10(0,0,0,0); TLorentzVector AK44_15(0,0,0,0);
     TLorentzVector AK44_20(0,0,0,0); 

     for(auto iP = AK41_parts.begin(); iP != AK41_parts.end(); iP++)
     {
         iP->Boost(-AK41_boost.X(),-AK41_boost.Y(), -AK41_boost.Z());
         boostedAK41_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
         boostedAK41_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
         boostedAK41_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() ));

         if(iP->E() > 0.1)
         {
            AK41_0p1+= *iP;
            n_AK41_parts_0p1++;
         }
         if(iP->E() > 0.5)
         {
            AK41_0p5+= *iP;
            n_AK41_parts_0p5++;
         }
         if(iP->E() > 1)
         {
            AK41_1+= *iP;
            n_AK41_parts_1++;
         }
         if(iP->E() > 2)
         {
            AK41_2+= *iP;
            n_AK41_parts_2++;
         }
         if(iP->E() > 5)
         {
            AK41_5+= *iP;
            n_AK41_parts_5++;
         }
         if(iP->E() > 7.5)
         {
            AK41_7p5+= *iP;
            n_AK41_parts_7p5++;
         }
         if(iP->E() > 10)
         {
            AK41_10+= *iP;
            n_AK41_parts_10++;
         }
         if(iP->E() > 15)
         {
            AK41_15+= *iP;
            n_AK41_parts_15++;
         }
         sumPz_AK41+= abs(iP->Pz()); 
         sumP_AK41 += abs(iP->P());
     }
     for(auto iP = AK42_parts.begin(); iP != AK42_parts.end(); iP++)
     {
         iP->Boost(-AK42_boost.X(),-AK42_boost.Y(), -AK42_boost.Z());
         boostedAK42_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
         boostedAK42_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
         boostedAK42_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() ));  
         if(iP->E() > 0.1)
         {
            AK42_0p1+= *iP;
            n_AK42_parts_0p1++;
         }
         if(iP->E() > 0.5)
         {
            AK42_0p5+= *iP;
            n_AK42_parts_0p5++;
         }
         if(iP->E() > 1)
         {
            AK42_1+= *iP;
            n_AK42_parts_1++;
         }
         if(iP->E() > 2)
         {
            AK42_2+= *iP;
            n_AK42_parts_2++;
         }
         if(iP->E() > 5)
         {
            AK42_5+= *iP;
            n_AK42_parts_5++;
         }
         if(iP->E() > 7.5)
         {
            AK42_7p5+= *iP;
            n_AK42_parts_7p5++;
         }
         if(iP->E() > 10)
         {
            AK42_10+= *iP;
            n_AK42_parts_10++;
         }
         if(iP->E() > 15)
         {
            AK42_15+= *iP;
            n_AK42_parts_15++;
         }
         sumPz_AK42+= abs(iP->Pz()); 
         sumP_AK42 += abs(iP->P());
     }
     for(auto iP = AK43_parts.begin(); iP != AK43_parts.end(); iP++)
     {
         iP->Boost(-AK43_boost.X(),-AK43_boost.Y(), -AK43_boost.Z());
         boostedAK43_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
         boostedAK43_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
         boostedAK43_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() )); 
         if(iP->E() > 0.1)
         {
            AK43_0p1+= *iP;
            n_AK43_parts_0p1++;
         }
         if(iP->E() > 0.5)
         {
            AK43_0p5+= *iP;
            n_AK43_parts_0p5++;
         }
         if(iP->E() > 1)
         {
            AK43_1+= *iP;
            n_AK43_parts_1++;
         }
         if(iP->E() > 2)
         {
            AK43_2+= *iP;
            n_AK43_parts_2++;
         }
         if(iP->E() > 5)
         {
            AK43_5+= *iP;
            n_AK43_parts_5++;
         }
         if(iP->E() > 7.5)
         {
            AK43_7p5+= *iP;
            n_AK43_parts_7p5++;
         }
         if(iP->E() > 10)
         {
            AK43_10+= *iP;
            n_AK43_parts_10++;
         }
         if(iP->E() > 15)
         {
            AK43_15+= *iP;
            n_AK43_parts_15++;
         }
         sumPz_AK43+= abs(iP->Pz()); 
         sumP_AK43 += abs(iP->P());
     }
     for(auto iP = AK44_parts.begin(); iP != AK44_parts.end(); iP++)
     {
         iP->Boost(-AK44_boost.X(),-AK44_boost.Y(), -AK44_boost.Z());
         boostedAK44_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
         boostedAK44_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
         boostedAK44_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() )); 
         if(iP->E() > 0.1)
         {
            AK44_0p1+= *iP;
            n_AK44_parts_0p1++;
         }
         if(iP->E() > 0.5)
         {
            AK44_0p5+= *iP;
            n_AK44_parts_0p5++;
         }
         if(iP->E() > 1)
         {
            AK44_1+= *iP;
            n_AK44_parts_1++;
         }
         if(iP->E() > 2)
         {
            AK44_2+= *iP;
            n_AK44_parts_2++;
         }
         if(iP->E() > 5)
         {
            AK44_5+= *iP;
            n_AK44_parts_5++;
         }
         if(iP->E() > 7.5)
         {
            AK44_7p5+= *iP;
            n_AK44_parts_7p5++;
         }
         if(iP->E() > 10)
         {
            AK44_10+= *iP;
            n_AK44_parts_10++;
         }
         if(iP->E() > 15)
         {
            AK44_15+= *iP;
            n_AK44_parts_15++;
         }
         sumPz_AK44+= abs(iP->Pz()); 
         sumP_AK44 += abs(iP->P());
     }

     ////vectors to get angles

     TVector3 AK4_jet1(jetsFJ_jet[0].px(),jetsFJ_jet[0].py(),jetsFJ_jet[0].pz());
     TVector3 AK4_jet2(jetsFJ_jet[1].px(),jetsFJ_jet[1].py(),jetsFJ_jet[1].pz());
     TVector3 AK4_jet3(jetsFJ_jet[2].px(),jetsFJ_jet[2].py(),jetsFJ_jet[2].pz());
     TVector3 AK4_jet4(jetsFJ_jet[3].px(),jetsFJ_jet[3].py(),jetsFJ_jet[3].pz());

     //fill superjet variables here ...
     //SJ mass variables

     if(debug)std::cout << "Calculating threshold CA4 combined masses." << std::endl;

     treeVars["tot_HT"] = totHT;  
     treeVars["eventNumber"] = eventNumber;
     treeVars["SJ_mass"]   = sqrt(pow(superJetE,2)-pow(superJetpx,2)-pow(superJetpy,2)-pow(superJetpz,2)); 
     treeVars["SJ_mass_25"] = sqrt(pow(SJ_25_E,2)-pow(SJ_25_px,2)-pow(SJ_25_py,2)-pow(SJ_25_pz,2)); 
     treeVars["SJ_mass_50"] = sqrt(pow(SJ_50_E,2)-pow(SJ_50_px,2)-pow(SJ_50_py,2)-pow(SJ_50_pz,2)); 
     treeVars["SJ_mass_100"] = sqrt(pow(SJ_100_E,2)-pow(SJ_100_px,2)-pow(SJ_100_py,2)-pow(SJ_100_pz,2)); 
     treeVars["SJ_mass_150"] = sqrt(pow(SJ_150_E,2)-pow(SJ_150_px,2)-pow(SJ_150_py,2)-pow(SJ_150_pz,2)); 
     treeVars["SJ_mass_200"] = sqrt(pow(SJ_200_E,2)-pow(SJ_200_px,2)-pow(SJ_200_py,2)-pow(SJ_200_pz,2)); 
     treeVars["SJ_mass_300"] = sqrt(pow(SJ_300_E,2)-pow(SJ_300_px,2)-pow(SJ_300_py,2)-pow(SJ_300_pz,2)); 
     
     treeVars["SJ_mass_400"] = sqrt(pow(SJ_400_E,2)-pow(SJ_400_px,2)-pow(SJ_400_py,2)-pow(SJ_400_pz,2)); 
     treeVars["SJ_mass_500"] = sqrt(pow(SJ_500_E,2)-pow(SJ_500_px,2)-pow(SJ_500_py,2)-pow(SJ_500_pz,2)); 
     treeVars["SJ_mass_800"] = sqrt(pow(SJ_800_E,2)-pow(SJ_800_px,2)-pow(SJ_800_py,2)-pow(SJ_800_pz,2)); 
     treeVars["SJ_mass_1000"] = sqrt(pow(SJ_1000_E,2)-pow(SJ_1000_px,2)-pow(SJ_1000_py,2)-pow(SJ_1000_pz,2)); 
     double offsetInts = 0.5;

     //SJ nAK4 variables
     if(debug)std::cout << "Calculating the nAK4_XXX vars." << std::endl;

     treeVars["SJ_nAK4_25"] = SJ_nAK4_25_ + offsetInts; 
     treeVars["SJ_nAK4_50"] = SJ_nAK4_50_ + offsetInts; 
     treeVars["SJ_nAK4_100"] = SJ_nAK4_100_ + offsetInts; 
     treeVars["SJ_nAK4_150"] = SJ_nAK4_150_ + offsetInts; 
     treeVars["SJ_nAK4_200"] = SJ_nAK4_200_ + offsetInts; 
     treeVars["SJ_nAK4_300"] = SJ_nAK4_300_ + offsetInts; 
     treeVars["SJ_nAK4_400"] = SJ_nAK4_400_ + offsetInts; 
     treeVars["SJ_nAK4_500"] = SJ_nAK4_500_ + offsetInts; 
     treeVars["SJ_nAK4_800"] = SJ_nAK4_800_ + offsetInts; 
     treeVars["SJ_nAK4_1000"] = SJ_nAK4_1000_ + offsetInts; 

     treeVars["AK41_nDaughters"] = jetsFJ_jet[0].constituents().size() + offsetInts; 
     treeVars["AK42_nDaughters"] = jetsFJ_jet[1].constituents().size() + offsetInts; 
     treeVars["AK43_nDaughters"] = jetsFJ_jet[2].constituents().size() + offsetInts; 
     treeVars["AK44_nDaughters"] = jetsFJ_jet[3].constituents().size() + offsetInts; 


     //softdrop mass???
     if(debug)std::cout << "Calculating CA4 mass combinations." << std::endl;

     //AK4 jet mass combinations
     treeVars["AK4_m1"] = jetsFJ_jet[0].m();   
     treeVars["AK4_m2"] = jetsFJ_jet[1].m(); 
     treeVars["AK4_m3"] = jetsFJ_jet[2].m(); 
     treeVars["AK4_m4"] = jetsFJ_jet[3].m(); 

     treeVars["AK41_E"] = jetsFJ_jet[0].E(); 
     treeVars["AK42_E"] = jetsFJ_jet[1].E(); 
     treeVars["AK43_E"] = jetsFJ_jet[2].E(); 
     treeVars["AK44_E"] = jetsFJ_jet[3].E(); 

     treeVars["AK41_px"] = jetsFJ_jet[0].px(); 
     treeVars["AK42_px"] = jetsFJ_jet[1].px(); 
     treeVars["AK43_px"] = jetsFJ_jet[2].px(); 
     treeVars["AK44_px"] = jetsFJ_jet[3].px(); 

     treeVars["AK41_py"] = jetsFJ_jet[0].py(); 
     treeVars["AK42_py"] = jetsFJ_jet[1].py(); 
     treeVars["AK43_py"] = jetsFJ_jet[2].py(); 
     treeVars["AK44_py"] = jetsFJ_jet[3].py(); 

     treeVars["AK41_pz"] = jetsFJ_jet[0].pz(); 
     treeVars["AK42_pz"] = jetsFJ_jet[1].pz(); 
     treeVars["AK43_pz"] = jetsFJ_jet[2].pz(); 
     treeVars["AK44_pz"] = jetsFJ_jet[3].pz(); 

     treeVars["AK4_m12"] = sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[1].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[1].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[1].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[1].pz(),2));   
     treeVars["AK4_m13"] = sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[2].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[2].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[2].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[2].pz(),2));   
     treeVars["AK4_m14"] = sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[3].pz(),2));   
     treeVars["AK4_m23"] = sqrt( pow(jetsFJ_jet[2].E() + jetsFJ_jet[1].E() ,2) - pow(jetsFJ_jet[2].px() + jetsFJ_jet[1].px(),2) - pow(jetsFJ_jet[2].py() + jetsFJ_jet[1].py(),2)- pow(jetsFJ_jet[2].pz() + jetsFJ_jet[1].pz(),2));   
     treeVars["AK4_m24"] = sqrt( pow(jetsFJ_jet[3].E() + jetsFJ_jet[1].E() ,2) - pow(jetsFJ_jet[3].px() + jetsFJ_jet[1].px(),2) - pow(jetsFJ_jet[3].py() + jetsFJ_jet[1].py(),2)- pow(jetsFJ_jet[3].pz() + jetsFJ_jet[1].pz(),2));   
     treeVars["AK4_m34"] = sqrt( pow(jetsFJ_jet[2].E() + jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[2].px() + jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[2].py() + jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[2].pz() + jetsFJ_jet[3].pz(),2));   

     treeVars["AK4_m123"] =sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[1].E() + jetsFJ_jet[2].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[1].px() + jetsFJ_jet[2].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[1].py() + jetsFJ_jet[2].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[1].pz() + jetsFJ_jet[2].pz(),2));   
     treeVars["AK4_m124"] =sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[1].E() + jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[1].px() + jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[1].py() + jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[1].pz() + jetsFJ_jet[3].pz(),2));   
     treeVars["AK4_m134"] =sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[2].E() + jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[2].px() + jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[2].py() + jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[2].pz() + jetsFJ_jet[3].pz(),2));   
     treeVars["AK4_m234"] =sqrt( pow(jetsFJ_jet[1].E() + jetsFJ_jet[2].E() + jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[1].px() + jetsFJ_jet[2].px() + jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[1].py() + jetsFJ_jet[2].py() + jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[1].pz() + jetsFJ_jet[2].pz() + jetsFJ_jet[3].pz(),2));   

     treeVars["AK4_m1234"] =sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[1].E() + jetsFJ_jet[2].E()+jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[1].px() + jetsFJ_jet[2].px()+jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[1].py() + jetsFJ_jet[2].py()+jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[1].pz() + jetsFJ_jet[2].pz()+jetsFJ_jet[3].pz(),2));   
   

     //AK4 jet angles 
     if(debug)std::cout << "Calculating CA4 jet angles." << std::endl;

     treeVars["AK4_theta12"] = cos(abs(AK4_jet1.Angle(AK4_jet2)));   
     treeVars["AK4_theta13"] = cos(abs(AK4_jet1.Angle(AK4_jet3)));
     treeVars["AK4_theta14"] = cos(abs(AK4_jet1.Angle(AK4_jet4)));
     treeVars["AK4_theta23"] = cos(abs(AK4_jet2.Angle(AK4_jet3)));
     treeVars["AK4_theta24"] = cos(abs(AK4_jet2.Angle(AK4_jet4)));
     treeVars["AK4_theta34"] = cos(abs(AK4_jet3.Angle(AK4_jet4)));

     if(debug)std::cout << "Calculating Thrust, FW, etc." << std::endl;

     EventShapeVariables eventShapesAK41( boostedAK41_Part_XYZ );
     Thrust thrustCalculatorAK41( boostedAK41_Part_LC.begin(), boostedAK41_Part_LC.end() );
     double fwmAK41[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
     FWMoments( boostedAK41_Part_TLV, fwmAK41); 

     EventShapeVariables eventShapesAK42( boostedAK42_Part_XYZ );
     Thrust thrustCalculatorAK42( boostedAK42_Part_LC.begin(), boostedAK42_Part_LC.end() );
     double fwmAK42[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
     FWMoments( boostedAK42_Part_TLV, fwmAK42); 

     EventShapeVariables eventShapesAK43( boostedAK43_Part_XYZ );
     Thrust thrustCalculatorAK43( boostedAK43_Part_LC.begin(), boostedAK43_Part_LC.end() );
     double fwmAK43[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
     FWMoments( boostedAK43_Part_TLV, fwmAK43); 

     EventShapeVariables eventShapesAK44( boostedAK44_Part_XYZ );
     Thrust thrustCalculatorAK44( boostedAK44_Part_LC.begin(), boostedAK44_Part_LC.end() );
     double fwmAK44[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
     FWMoments( boostedAK44_Part_TLV, fwmAK44); 

     if(debug)std::cout << "Calculating the rest of the BES variables." << std::endl;

     //AK4 jet boosted information - boost reclustered AK4 jets into their COM and look at BES variables, ndaughters, nsubjettiness
     treeVars["AK41_ndaughters"] = jetsFJ_jet[0].constituents().size() + offsetInts; 
     treeVars["AK41_nsubjets"] = jetsFJ_jet[0].n_exclusive_subjets(0.2) + offsetInts; 
     treeVars["AK41_thrust"] = thrustCalculatorAK41.thrust();
     treeVars["AK41_sphericity"] = eventShapesAK41.sphericity();


     treeVars["AK41_asymmetry"] = sumPz_AK41/ sumP_AK41;     //jetsFJ_jet[0].p();   // asymmetry should be calculated in the SJ frame, arbitrary ax
     treeVars["AK41_isotropy"] = eventShapesAK41.isotropy();
     treeVars["AK41_aplanarity"] = eventShapesAK41.aplanarity();
     treeVars["AK41_FW1"] = fwmAK41[1]; 
     treeVars["AK41_FW2"] = fwmAK41[2]; 
     treeVars["AK41_FW3"] = fwmAK41[3]; 
     treeVars["AK41_FW4"] = fwmAK41[4]; 

     treeVars["AK42_ndaughters"] = jetsFJ_jet[1].constituents().size() + offsetInts; 
     treeVars["AK42_nsubjets"] = jetsFJ_jet[1].exclusive_subjets(0.2).size() + offsetInts;
     treeVars["AK42_thrust"] = thrustCalculatorAK42.thrust(); 
     treeVars["AK42_sphericity"] = eventShapesAK42.sphericity();
     treeVars["AK42_asymmetry"] = sumPz_AK42/ sumP_AK42;   //jetsFJ_jet[1].p(); 
     treeVars["AK42_isotropy"] = eventShapesAK42.isotropy();
     treeVars["AK42_aplanarity"] = eventShapesAK42.aplanarity();
     treeVars["AK42_FW1"] = fwmAK42[1]; 
     treeVars["AK42_FW2"] = fwmAK42[2]; 
     treeVars["AK42_FW3"] = fwmAK42[3]; 
     treeVars["AK42_FW4"] = fwmAK42[4]; 

     treeVars["AK43_ndaughters"] = jetsFJ_jet[2].constituents().size() + offsetInts; 
     treeVars["AK43_nsubjets"] = jetsFJ_jet[2].exclusive_subjets(0.2).size() + offsetInts;
     treeVars["AK43_thrust"] = thrustCalculatorAK43.thrust();
     treeVars["AK43_sphericity"] = eventShapesAK43.sphericity();
     treeVars["AK43_asymmetry"] = sumPz_AK43/ sumP_AK43;  //jetsFJ_jet[2].p(); 
     treeVars["AK43_isotropy"] = eventShapesAK43.isotropy();
     treeVars["AK43_aplanarity"] = eventShapesAK43.aplanarity();
     treeVars["AK43_FW1"] = fwmAK43[1]; 
     treeVars["AK43_FW2"] = fwmAK43[2]; 
     treeVars["AK43_FW3"] = fwmAK43[3]; 
     treeVars["AK43_FW4"] = fwmAK43[4]; 

     treeVars["AK44_ndaughters"] = jetsFJ_jet[3].constituents().size() + offsetInts; 
     treeVars["AK44_nsubjets"] = jetsFJ_jet[3].exclusive_subjets(0.2).size() + offsetInts;
     treeVars["AK44_thrust"] = thrustCalculatorAK44.thrust();
     treeVars["AK44_sphericity"] = eventShapesAK44.sphericity();

     treeVars["AK44_asymmetry"] = sumPz_AK44 / sumP_AK44;  //jetsFJ_jet[3].p(); 
     treeVars["AK44_isotropy"] = eventShapesAK44.isotropy();
     treeVars["AK44_aplanarity"] = eventShapesAK44.aplanarity();
     treeVars["AK44_FW1"] = fwmAK44[1]; 
     treeVars["AK44_FW2"] = fwmAK44[2]; 
     treeVars["AK44_FW3"] = fwmAK44[3]; 
     treeVars["AK44_FW4"] = fwmAK44[4]; 

     //Full SJ BES variablesf
     EventShapeVariables eventShapes( boostedSuperJetPart_XYZ );
     Thrust thrustCalculator( boostedSuperJetPart_LC.begin(), boostedSuperJetPart_LC.end() );
     double fwm[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
     FWMoments( boostedSuperJetPart_TLV, fwm); 
     treeVars["SJ_thrust"] = thrustCalculator.thrust();
     treeVars["SJ_sphericity"] = eventShapes.sphericity();
     treeVars["SJ_asymmetry"] = sumPz/sumP; 
     treeVars["SJ_isotropy"] = eventShapes.isotropy();
     treeVars["SJ_aplanarity"] = eventShapes.aplanarity();
     treeVars["SJ_FW1"] = fwm[1]; 
     treeVars["SJ_FW2"] = fwm[2]; 
     treeVars["SJ_FW3"] = fwm[3]; 
     treeVars["SJ_FW4"] = fwm[4]; 


     if(testNewVars)
     {

        treeVars["SJ_AK4_frac_10"] = 1.0*SJ_nAK4_10_ / SJ_nAK4_1;
        treeVars["SJ_AK4_frac_25"] = 1.0*SJ_nAK4_25_ / SJ_nAK4_1;
        treeVars["SJ_AK4_frac_50"] = 1.0*SJ_nAK4_50_ / SJ_nAK4_1;
        treeVars["SJ_AK4_frac_75"] = 1.0*SJ_nAK4_75_ / SJ_nAK4_1;
        treeVars["SJ_AK4_frac_100"] = 1.0*SJ_nAK4_100_/ SJ_nAK4_1;
        treeVars["SJ_AK4_frac_200"] = 1.0*SJ_nAK4_200_/ SJ_nAK4_1;
        treeVars["SJ_AK4_frac_300"] = 1.0*SJ_nAK4_300_/ SJ_nAK4_1;
        treeVars["SJ_AK4_frac_500"] = 1.0*SJ_nAK4_500_ / SJ_nAK4_1;
        treeVars["SJ_AK4_frac_800"] = 1.0*SJ_nAK4_800_/ SJ_nAK4_1;

        treeVars["AK41_daughters_frac_0p1"] = 1.0*n_AK41_parts_0p1  /AK41_parts.size();   
        treeVars["AK41_daughters_frac_0p5"] = 1.0*n_AK41_parts_0p5  /AK41_parts.size();   
        treeVars["AK41_daughters_frac_1"] = 1.0*n_AK41_parts_1  /AK41_parts.size();   
        treeVars["AK41_daughters_frac_2"] = 1.0*n_AK41_parts_2  /AK41_parts.size();   
        treeVars["AK41_daughters_frac_5"] = 1.0*n_AK41_parts_5  /AK41_parts.size(); 
        treeVars["AK41_daughters_frac_7p5"] = 1.0*n_AK41_parts_7p5 /AK41_parts.size();     
        treeVars["AK41_daughters_frac_10"] = 1.0*n_AK41_parts_10  /AK41_parts.size();   
        treeVars["AK41_daughters_frac_15"] = 1.0*n_AK41_parts_15 /AK41_parts.size();   

        treeVars["AK41_mass_0p1"] = AK41_0p1.M();
        treeVars["AK41_mass_0p5"] = AK41_0p5.M();
        treeVars["AK41_mass_1"] = AK41_1.M();
        treeVars["AK41_mass_2"] = AK41_2.M();
        treeVars["AK41_mass_5"] = AK41_5.M();
        treeVars["AK41_mass_7p5"] = AK41_7p5.M();
        treeVars["AK41_mass_10"] = AK41_10.M();
        treeVars["AK41_mass_15"] = AK41_15.M();

        treeVars["AK42_daughters_frac_0p1"] = 1.0*n_AK42_parts_0p1  /AK42_parts.size();   
        treeVars["AK42_daughters_frac_0p5"] = 1.0*n_AK42_parts_0p5  /AK42_parts.size();   
        treeVars["AK42_daughters_frac_1"] = 1.0*n_AK42_parts_1  /AK42_parts.size();   
        treeVars["AK42_daughters_frac_2"] = 1.0*n_AK42_parts_2  /AK42_parts.size();   
        treeVars["AK42_daughters_frac_5"] = 1.0*n_AK42_parts_5  /AK42_parts.size(); 
        treeVars["AK42_daughters_frac_7p5"] = 1.0*n_AK42_parts_7p5 /AK42_parts.size();     
        treeVars["AK42_daughters_frac_10"] = 1.0*n_AK42_parts_10  /AK42_parts.size();   
        treeVars["AK42_daughters_frac_15"] = 1.0*n_AK42_parts_15 /AK42_parts.size();   

        treeVars["AK42_mass_0p1"] = AK42_0p1.M();
        treeVars["AK42_mass_0p5"] = AK42_0p5.M();
        treeVars["AK42_mass_1"] = AK42_1.M();
        treeVars["AK42_mass_2"] = AK42_2.M();
        treeVars["AK42_mass_5"] = AK42_5.M();
        treeVars["AK42_mass_7p5"] = AK42_7p5.M();
        treeVars["AK42_mass_10"] = AK42_10.M();
        treeVars["AK42_mass_15"] = AK42_15.M();

        treeVars["AK43_daughters_frac_0p1"] = 1.0*n_AK43_parts_0p1  /AK43_parts.size();   
        treeVars["AK43_daughters_frac_0p5"] = 1.0*n_AK43_parts_0p5  /AK43_parts.size();   
        treeVars["AK43_daughters_frac_1"] = 1.0*n_AK43_parts_1  /AK43_parts.size();   
        treeVars["AK43_daughters_frac_2"] = 1.0*n_AK43_parts_2  /AK43_parts.size();   
        treeVars["AK43_daughters_frac_5"] = 1.0*n_AK43_parts_5  /AK43_parts.size(); 
        treeVars["AK43_daughters_frac_7p5"] = 1.0*n_AK43_parts_7p5 /AK43_parts.size();     
        treeVars["AK43_daughters_frac_10"] = 1.0*n_AK43_parts_10  /AK43_parts.size();   
        treeVars["AK43_daughters_frac_15"] = 1.0*n_AK43_parts_15 /AK43_parts.size();   

        treeVars["AK43_mass_0p1"] = AK43_0p1.M();
        treeVars["AK43_mass_0p5"] = AK43_0p5.M();
        treeVars["AK43_mass_1"] = AK43_1.M();
        treeVars["AK43_mass_2"] = AK43_2.M();
        treeVars["AK43_mass_5"] = AK43_5.M();
        treeVars["AK43_mass_7p5"] = AK43_7p5.M();
        treeVars["AK43_mass_10"] = AK43_10.M();
        treeVars["AK43_mass_15"] = AK43_15.M();

        treeVars["AK44_daughters_frac_0p1"] = 1.0*n_AK44_parts_0p1  /AK44_parts.size();   
        treeVars["AK44_daughters_frac_0p5"] = 1.0*n_AK44_parts_0p5  /AK44_parts.size();   
        treeVars["AK44_daughters_frac_1"] = 1.0*n_AK44_parts_1  /AK44_parts.size();   
        treeVars["AK44_daughters_frac_2"] = 1.0*n_AK44_parts_2  /AK44_parts.size();   
        treeVars["AK44_daughters_frac_5"] = 1.0*n_AK44_parts_5  /AK44_parts.size(); 
        treeVars["AK44_daughters_frac_7p5"] = 1.0*n_AK44_parts_7p5 /AK44_parts.size();     
        treeVars["AK44_daughters_frac_10"] = 1.0*n_AK44_parts_10  /AK44_parts.size();   
        treeVars["AK44_daughters_frac_15"] = 1.0*n_AK44_parts_15 /AK44_parts.size();   

        treeVars["AK44_mass_0p1"] = AK44_0p1.M();
        treeVars["AK44_mass_0p5"] = AK44_0p5.M();
        treeVars["AK44_mass_1"] = AK44_1.M();
        treeVars["AK44_mass_2"] = AK44_2.M();
        treeVars["AK44_mass_5"] = AK44_5.M();
        treeVars["AK44_mass_7p5"] = AK44_7p5.M();
        treeVars["AK44_mass_10"] = AK44_10.M();
        treeVars["AK44_mass_15"] = AK44_15.M();

     }

     if(debug)std::cout << "Checking health of variables." << std::endl;

     for (unsigned i = 0; i < listOfVars.size(); i++)
     {

      if (  (treeVars[ listOfVars[i] ] != treeVars[ listOfVars[i] ] ) || ( isinf(treeVars[ listOfVars[i] ])   )  )
      {
         if(debug)std::cout << "Bad variable: " << listOfVars[i] << ". Setting to 0 " << std::endl;
         treeVars[ listOfVars[i] ] = 0.;
      }
      else if ( abs(treeVars[ listOfVars[i] ]+999.99 ) < 1.0e-10 )
      {
         std::cout << "Variable " << listOfVars[i] << " was not set" << std::endl;
         return;
      } 
     }

     if(debug2)std::cout << "Filling tree." << std::endl;

     superjetTree->Fill();


     /*
      std::cout << "-------------------------------- new SJ ----------------------------------------- " << std::endl;
      std::cout << "eventNumber" << " " << eventNumber << std::endl;
      std::cout << "superjet_num" << " " << nSuperJets + 1<< std::endl;
      std::cout << "totHT" << " " << totHT << std::endl;
      */
      for (unsigned i = 0; i < listOfVars.size(); i++)
     {

         if ( (listOfVars[i] ==  "tot_HT" ) || (  listOfVars[i] == "eventNumber")) continue;
         
        // std::cout<< listOfVars[i] << " "  << treeVars[ listOfVars[i] ] << std::endl;

         treeVars[ listOfVars[i] ] = -999.99;
     }
   }


   if(debug)std::cout << "--------------------- Finished event --------------------"<< std::endl;

}
void
BESTProducer::beginStream(edm::StreamID)
{
}

//=================================================================================
// Method called once each job just after ending the event loop  ------------------
//=================================================================================

void
BESTProducer::endStream()
{
}
void
BESTProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
   //The following says we do not know what parameters are allowed so do no validation
   // Please change this to state exactly what you do use, even if it is no parameters
   edm::ParameterSetDescription desc;
   desc.setUnknown();
   descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BESTProducer);



//777887

//primary vertex 






