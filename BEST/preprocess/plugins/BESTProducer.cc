// -*- C++ -*-
//========================================================================================
// Package:    BEST/preprocess                  ---------------------------------------
// Class:      BESTProducer                     ---------------------------------------
//----------------------------------------------------------------------------------------
/**\class BESTProducer BESTProducer.cc BEST/preprocess/plugins/BESTProducer.cc
------------------------------------------------------------------------------------------
 Description: This class preprocesses MC samples so that they can be used with BEST ---
 -----------------------------------------------------------------------------------------
 Implementation:                                                                       ---
     This EDProducer is meant to be used with CMSSW_9_4_8                              ---
*/
//========================================================================================
// Authors:  Brendan Regnery, Justin Pilot, Reyer Band, Devin Taylor ---------------------
//         Created:  WED, 8 Aug 2018 21:00:28 GMT  ---------------------------------------
//========================================================================================
//////////////////////////////////////////////////////////////////////////////////////////


// system include files
#include <memory>
#include <thread>
#include <iostream>

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
//#include "DataFormats/VertexReco/interface/VertexFwd.h"
//#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
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

///////////////////////////////////////////////////////////////////////////////////
// Define a namespace -------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

namespace best {


    typedef reco::Candidate::PolarLorentzVector fourv;
    typedef math::XYZTLorentzVector LorentzVector;
    typedef math::XYZVector Vector;
    using namespace reco;

    // Create a mapping from the input jetColl to enum

}


///////////////////////////////////////////////////////////////////////////////////
// Class declaration --------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

class BESTProducer : public edm::stream::EDProducer<> 
{
    public:
        explicit BESTProducer(const edm::ParameterSet&);
        ~BESTProducer();
        bool isMatchedtoSJ(std::vector<TLorentzVector>, TLorentzVector);
        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
        double calcMPP(TLorentzVector );
        const reco::Candidate* parse_chain(const reco::Candidate*);
        bool isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF, int nfatjets);
        bool isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF);
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

        // Input variables
        bool storeDaughters;

        // Tree variables
        TTree *jetTree;
        std::map<std::string, float> treeVars;
        std::vector<std::string> listOfVars;
        std::map<std::string, std::vector<float> > jetVecVars;
        std::vector<std::string> listOfVecVars;
        std::map<std::string, std::array<std::array<std::array<float, 1>, 31>, 31> > imgVars;
        std::vector<std::string> listOfImgVars;
        int nEvents = 0;
        int nWbHt = 0;
        int nWbZt = 0;
        int nWbWb = 0;
        int nHtZt = 0;
        int nHtHt = 0;
        int nZtZt = 0;
        int nPassedEvents = 0;
        // Tokens
        //edm::EDGetTokenT<std::vector<pat::PackedCandidate> > pfCandsToken_;
        edm::EDGetTokenT<std::vector<pat::Jet> > ak8JetsToken_;
        //edm::EDGetTokenT<std::vector<pat::Jet> > ak4JetsToken_;
        edm::EDGetTokenT<std::vector<reco::GenParticle> > genPartToken_;
        edm::EDGetTokenT<std::vector<reco::VertexCompositePtrCandidate> > secVerticesToken_;
        edm::EDGetTokenT<std::vector<reco::Vertex> > verticesToken_;
        //edm::EDGetTokenT<std::vector<pat::Jet> > ak8CHSSoftDropSubjetsToken_;
        std::string jetType_;
        edm::EDGetTokenT<std::vector<pat::Jet>> jetToken_;
        //edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
        //edm::EDGetTokenT<bool> BadChCandFilterToken_;
        //edm::EDGetTokenT<bool> BadPFMuonFilterToken_;
};

///////////////////////////////////////////////////////////////////////////////////
// constants, enums and typedefs --------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
// static data member definitions -------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
// Constructors -------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

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
bool BESTProducer::isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF, int nfatjets)
{
   if ( (nfatjets < 2) &&  (abs(eta) > 2.4) ) return false;
   else if ( (nfatjets >= 3) && (abs(eta) > 1.5) ) return false;

   if ((NHF>0.9) || (NEMF>0.9) || (NumConst<1) || (CHF<0.) || (CHM<0) || (MUF > 0.8) || (CEMF > 0.8)) 
      {
         return false;
      }
   else{ return true;}

}
bool BESTProducer::isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF)
{
   if( (abs(eta) > 2.5)) return false;

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

//given a list of (boosted to specific SJ COM) particles coming from b-tagged AK8 subjets and the particles of a specific AK4 jet
//returns fraction of those AK4 jet particles that originated from B-tagged subjets
// returns 0 if num AK4 daughters < 5


BESTProducer::BESTProducer(const edm::ParameterSet& iConfig):
    storeDaughters (iConfig.getParameter<bool>("storeDaughters"))
{

    //------------------------------------------------------------------------------
    // Prepare TFile Service -------------------------------------------------------
    //------------------------------------------------------------------------------

    edm::Service<TFileService> fs;
    jetTree = fs->make<TTree>("jetTree","jetTree");
    jetType_ = iConfig.getParameter<std::string>("jetType");
    //------------------------------------------------------------------------------
    // Create tree variables and branches ------------------------------------------
    //------------------------------------------------------------------------------
    // listOfVars is the flat part of the TTree ------------------------------------
    //------------------------------------------------------------------------------

        // SJ COM variable information
        listOfVars.push_back("tot_pt");
        listOfVars.push_back("tot_HT");
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

        //softdrop mass???
        //bjet particle percentages

        //AK4 jet mass combinations


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
        listOfVars.push_back("AK41_ndaughters");
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

        listOfVars.push_back("AK42_ndaughters");
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

        listOfVars.push_back("AK43_ndaughters");
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

        //listOfVars.push_back("AK434_ndaughters");
        //listOfVars.push_back("AK43_Tau3");
        //listOfVars.push_back("AK43_Tau2");
        //listOfVars.push_back("AK43_Tau1");
        //listOfVars.push_back("AK43_Tau21");
        //listOfVars.push_back("AK44_nsubjets");
        //listOfVars.push_back("AK44_thrust");
        //listOfVars.push_back("AK44_sphericity");
        //listOfVars.push_back("AK44_asymmetry");
        //listOfVars.push_back("AK44_isotropy");
        //listOfVars.push_back("AK44_aplanarity");
        //listOfVars.push_back("AK44_FW1");
        //listOfVars.push_back("AK44_FW2");
        //listOfVars.push_back("AK44_FW3");
        //listOfVars.push_back("AK44_FW4");

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

        //listOfVars.push_back("AK41_bTagRatio");
        //listOfVars.push_back("AK42_bTagRatio");
        //listOfVars.push_back("AK43_bTagRatio");
        //listOfVars.push_back("AK44_bTagRatio");

        /*
        listOfVars.push_back("_htWb");
        listOfVars.push_back("_htZt");
        listOfVars.push_back("_ZtWb");
        listOfVars.push_back("_WbWb");
        listOfVars.push_back("_htht");
        listOfVars.push_back("_ZtZt");
        */

    // Make Branches for each variable
    for (unsigned i = 0; i < listOfVars.size(); i++)
    {
        treeVars[ listOfVars[i] ] = -999.99;
        jetTree->Branch( (listOfVars[i]).c_str() , &(treeVars[ listOfVars[i] ]), (listOfVars[i]+"/F").c_str() );
    }

    //------------------------------------------------------------------------------
    // Define input tags -----------------------------------------------------------
    //------------------------------------------------------------------------------

    // AK8 Jets
    edm::InputTag ak8JetsTag_;
    ak8JetsTag_ = edm::InputTag("slimmedJetsAK8", "", "PAT");
    // ak8JetsTag_ = edm::InputTag(inputJetColl_, "", "run"); // this may be needed as an option for 2016 mc
    ak8JetsToken_ = consumes<std::vector<pat::Jet> >(ak8JetsTag_);

    // Gen Particles
    edm::InputTag genPartTag_;
    genPartTag_ = edm::InputTag("prunedGenParticles", "", "PAT");
    genPartToken_ = consumes<std::vector<reco::GenParticle> >(genPartTag_);

    // Primary Vertices
    edm::InputTag verticesTag_;
    verticesTag_ = edm::InputTag("offlineSlimmedPrimaryVertices", "", "PAT");
    verticesToken_ = consumes<std::vector<reco::Vertex> >(verticesTag_);

    // Secondary Vertices
    edm::InputTag secVerticesTag_;
    secVerticesTag_ = edm::InputTag("slimmedSecondaryVertices", "", "PAT");
    secVerticesToken_ = consumes<std::vector<reco::VertexCompositePtrCandidate> >(secVerticesTag_);
    
    jetToken_    = consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jetCollection"));

}

///////////////////////////////////////////////////////////////////////////////////
// Destructor ---------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

BESTProducer::~BESTProducer()
{

    // do anything that needs to be done at destruction time
    // (eg. close files, deallocate, resources etc.)

}

///////////////////////////////////////////////////////////////////////////////////
// Member Functions ---------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

//=================================================================================
// Method called for each event ---------------------------------------------------
//=================================================================================

void BESTProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    using namespace fastjet;
    using namespace std;
    //------------------------------------------------------------------------------
    // Create miniAOD object collections -------------------------------------------
    //------------------------------------------------------------------------------

    // Find objects corresponding to the token and link to the handle
    Handle< std::vector<pat::Jet> > ak8JetsCollection;
    iEvent.getByToken(ak8JetsToken_, ak8JetsCollection);
    vector<pat::Jet> ak8Jets = *ak8JetsCollection.product();

    Handle< std::vector<reco::GenParticle> > genPartCollection;
    iEvent.getByToken(genPartToken_, genPartCollection);
    vector<reco::GenParticle> genPart = *genPartCollection.product();

    Handle< std::vector<reco::Vertex> > vertexCollection;
    iEvent.getByToken(verticesToken_, vertexCollection);
    vector<reco::Vertex> pVertices = *vertexCollection.product();

    Handle< std::vector<reco::VertexCompositePtrCandidate> > secVertexCollection;
    iEvent.getByToken(secVerticesToken_, secVertexCollection);
    vector<reco::VertexCompositePtrCandidate> secVertices = *secVertexCollection.product();


    //------------------------------------------------------------------------------
    // Gen Particles Loop ----------------------------------------------------------
    //------------------------------------------------------------------------------

   int chi_pdgid = 8000001;
   int nSuu = 0;
   int nChi = 0;
   int nW   = 0;
   int ntop = 0;
   int nH   = 0;
   int nhq  = 0;
   int nSuub = 0;
   int nZ    = 0;
   int nZq   = 0;
   int nTopb = 0;
   int nWq   = 0;
   int nhGlu = 0;
   int nhWq = 0;
   int nhZq = 0;

   std::vector<TLorentzVector> genChiZt;
   std::vector<TLorentzVector> genChiHt;
   std::vector<TLorentzVector> genChiWb;


   std::vector<TLorentzVector> Zquarks;
   std::vector<TLorentzVector> Wquarks;
   std::vector<TLorentzVector> Hquarks;
   std::vector<TLorentzVector> TopWquarks;
   std::vector<TLorentzVector> Topb;
   std::vector<TLorentzVector> Suub;

   
//want only fully hadronic events_
   for (auto iG = genPartCollection->begin(); iG != genPartCollection->end(); iG++) 
   {
      if ((abs(iG->pdgId()) == 24) && ((abs(iG->mother()->pdgId()) == chi_pdgid)) ) 
      {
         genChiWb.push_back( TLorentzVector(iG->mother()->px(),iG->mother()->py(),iG->mother()->pz(),iG->mother()->energy())  );
         const reco::Candidate* W_final = parse_chain(iG->clone());
         for (unsigned int iii=0; iii<W_final->numberOfDaughters(); iii++)
         {
            const reco::Candidate* W_daughter = W_final->daughter(iii);
            if(abs(W_daughter->pdgId()) < 6)
            { 
               Wquarks.push_back(TLorentzVector(W_daughter->px(),W_daughter->py(),W_daughter->pz(),W_daughter->energy()));
               nWq++;
            }
         }

         nW++;
      }
      else if ( (abs(iG->pdgId()) == 5) && (abs(iG->mother()->pdgId()) == chi_pdgid)  )
      {
         Suub.push_back(TLorentzVector(iG->px(),iG->py(),iG->pz(),iG->energy()));
         nSuub++;
      } 

      else if ( (abs(iG->pdgId()) == 6) && ((abs(iG->mother()->pdgId()) == chi_pdgid)) ) 
      {
         const reco::Candidate* t_final = parse_chain(iG->clone());

         for (unsigned int iii=0; iii<t_final->numberOfDaughters(); iii++)
         {
            const reco::Candidate* t_daughter = t_final->daughter(iii);
            if (abs(t_daughter->pdgId())==24) 
            {
               const reco::Candidate* W_final = parse_chain(t_daughter->clone());
               for (unsigned int jjj=0; jjj<W_final->numberOfDaughters(); jjj++)
               {
                  const reco::Candidate* W_daughter = W_final->daughter(jjj);
                  if(abs(W_daughter->pdgId()) < 6)
                  {
                     TopWquarks.push_back(TLorentzVector(W_daughter->px(),W_daughter->py(),W_daughter->pz(),W_daughter->energy()));
                     nWq++;
                  }
               }
               nW++;
            }
            else if(abs(t_daughter->pdgId())==5) 
            {
               Topb.push_back(TLorentzVector(t_daughter->px(),t_daughter->py(),t_daughter->pz(),t_daughter->energy()));
               nTopb++;
            }
         }
         ntop++;
      }
      else if ( (abs(iG->pdgId()) == 25) && ((abs(iG->mother()->pdgId()) == chi_pdgid)) ) 
      {
         genChiHt.push_back( TLorentzVector(iG->mother()->px(),iG->mother()->py(),iG->mother()->pz(),iG->mother()->energy())  );
         const reco::Candidate* h_final = parse_chain(iG->clone());
         for (unsigned int iii=0; iii<h_final->numberOfDaughters(); iii++)
         {
            const reco::Candidate* h_daughter = h_final->daughter(iii);
            if (abs(h_daughter->pdgId())<6)
            {
               Hquarks.push_back(TLorentzVector(h_daughter->px(),h_daughter->py(),h_daughter->pz(),h_daughter->energy()));
               nhq++;
            }
            else if (abs(h_daughter->pdgId()) ==24)
            {
               const reco::Candidate* higgsW_final = parse_chain(h_daughter->clone());
               for (unsigned int jjj=0; jjj<higgsW_final->numberOfDaughters(); jjj++)
               {
                  const reco::Candidate* higgsW_daughter = higgsW_final->daughter(jjj);
                  if( (abs(higgsW_daughter->pdgId()) < 6))
                  {
                     Hquarks.push_back(TLorentzVector(h_daughter->px(),h_daughter->py(),h_daughter->pz(),h_daughter->energy()));
                     nhWq++;
                  } 
               }
            }
            else if (abs(h_daughter->pdgId()) ==23)
            {
               const reco::Candidate* higgsZ_final = parse_chain(h_daughter->clone());
               for (unsigned int jjj=0; jjj<higgsZ_final->numberOfDaughters(); jjj++)
               {
                  const reco::Candidate* higgsZ_daughter = higgsZ_final->daughter(jjj);
                  if( (abs(higgsZ_daughter->pdgId()) < 6))
                  {
                     Hquarks.push_back(TLorentzVector(h_daughter->px(),h_daughter->py(),h_daughter->pz(),h_daughter->energy()));
                     nhZq++;
                  } 
               }
            }
            else if (abs(h_daughter->pdgId()) ==21)
            {
                Hquarks.push_back(TLorentzVector(h_daughter->px(),h_daughter->py(),h_daughter->pz(),h_daughter->energy()));
                nhGlu++;
            }
         }
         nH++;
      }
      else if ( (abs(iG->pdgId()) == 23) && ((abs(iG->mother()->pdgId()) == chi_pdgid)) ) 
      {
         genChiZt.push_back( TLorentzVector(iG->mother()->px(),iG->mother()->py(),iG->mother()->pz(),iG->mother()->energy())  );

         const reco::Candidate* Z_final = parse_chain(iG->clone());
         for (unsigned int jjj=0; jjj<Z_final->numberOfDaughters(); jjj++)
         {
            const reco::Candidate* Z_daughter = Z_final->daughter(jjj);
            if(abs(Z_daughter->pdgId()) < 6)
            {

               Zquarks.push_back(TLorentzVector(Z_daughter->px(),Z_daughter->py(),Z_daughter->pz(),Z_daughter->energy()));
               nZq++;
            }         
         }
         nZ++;

      }
      else if ((abs(iG->pdgId()) == chi_pdgid) && (iG->isLastCopy()))
      {
         nChi++;
      }    
      //else if  ((abs(iG->pdgId()) == chi_pdgid) && (iG->isLastCopy())) nChi++;
   }


   bool _htWb = false;
   bool _htZt = false;
   bool _ZtWb = false;
   bool _WbWb = false;
   bool _htht = false;
   bool _ZtZt = false;
   nEvents++;

   //if      ( (nChi<2) || (nSuu < 1) ) return; //don't want anything to do with these bad events anyways 
   if      ((nH == 2) && (nSuub == 0) && (nTopb >= 2) && ( (nhq+nhGlu+nhWq+nhZq) == 4) && (nWq == 4) && (nZq == 0) && (ntop == 2) && (nW == 2)  && (nZ == 0))
   {
     // _htht = true; 
      nHtHt++;
   } 
   else if ((nH == 0) && (nSuub == 0) && (nTopb >= 2) && ( (nhq+nhGlu+nhWq+nhZq) == 0) && (nWq == 4) && (nZq == 4) && (ntop == 2) && (nW == 2)  && (nZ == 2))
   {
    //  _ZtZt = true;
      nZtZt++;
   } 
   else if ((nH == 0) && (nSuub == 2) && (nTopb >= 0) && ( (nhq+nhGlu+nhWq+nhZq) == 0) && (nWq == 4) && (nZq == 0) && (ntop == 0) && (nW == 2)  && (nZ == 0))
   {
    // _WbWb = true;
      nWbWb++;
   } 
   else if ((nH == 1) && (nSuub == 1) && (nTopb >= 1) && ( (nhq+nhGlu+nhWq+nhZq) == 2) && (nWq == 4) && (nZq == 0) && (ntop == 1) && (nW == 2)  && (nZ == 0))
   {
     // _htWb = true;
      nWbHt++;
   }
   else if ((nH == 0) && (nSuub == 1) && (nTopb >= 1) && ( (nhq+nhGlu+nhWq+nhZq) == 0) && (nWq == 4) && (nZq == 2) && (ntop == 1) && (nW == 2)  && (nZ == 1))
   {
      _ZtWb = true;
      nWbZt++;
   }
   else if ((nH == 1) && (nSuub == 0) && (nTopb >= 2) && ( (nhq+nhGlu+nhWq+nhZq) == 2) && (nWq == 4) && (nZq == 2) && (ntop == 2) && (nW == 2)  && (nZ == 1))
   { 
      _htZt = true;
      nHtZt++;
   }   
   //std::cout << "nHtHt " << nHtHt<< " nWbWb "<< nWbWb<< " nZtZt " << nZtZt<< " nWbHt " << nWbHt<< " nWbZt " << nWbZt<< " nHtZt" << nHtZt << std::endl;
/* 
   if      ((nH == 2) && (nSuub >= 0) && (ntop == 2) && (nW == 2)  && (nZ == 0))
   {
      _htht = true; 
   } 
   else if ((nH == 0) && (nSuub >= 0) && (ntop == 2) && (nW == 2)  && (nZ == 2))
   {
      _ZtZt = true;
   } 
   else if ((nH == 0) && (nSuub >= 2) && (ntop == 0) && (nW == 2)  && (nZ == 0))
   {
      _WbWb = true;
   } 
   else if ((nH == 1) && (nSuub >= 1) && (ntop == 1) && (nW == 2)  && (nZ == 0))
   {
      _htWb = true;
   }
   else if ((nH == 0) && (nSuub >= 1) && (ntop == 1) && (nW == 2)  && (nZ == 1))
   {
      _ZtWb = true;
   }
   else if ((nH == 1) && (nSuub >= 0) && (ntop == 2) && (nW == 2)  && (nZ == 1))
   {
      _htZt = true;
   }  
*/
   if (! ( _htWb || _htZt || _ZtWb || _WbWb || _htht || _ZtZt)  ) return;
   //std::cout << "nHtHt " << nHtHt<< " nWbWb "<< nWbWb<< " nZtZt " << nZtZt<< " nWbHt " << nWbHt<< " nWbZt " << nWbZt<< " nHtZt" << nHtZt << std::endl;
    //------------------------------------------------------------------------------
    // AK8 Jet Loop ----------------------------------------------------------------
    //------------------------------------------------------------------------------
    // This loop makes a tree entry for each jet of interest -----------------------
    //------------------------------------------------------------------------------

    // Create structures for storing daughters and rest frame jets
    vector<reco::Candidate * > daughtersOfJet;
    map<string, vector<TLorentzVector> > boostedDaughters;
    map<string, vector<fastjet::PseudoJet> > restJets;
   //std::vector<std::pair <TLorentzVector,double>> particlesPUPPI; 
   std::vector<reco::LeafCandidate> candsBoosted;   
   std::vector<reco::LeafCandidate> candsUnboosted;



////////////////////////////////////AK4 jets/////////////////////////////////////////////////////////


   edm::Handle<std::vector<pat::Jet> > smallJets;
   iEvent.getByToken(jetToken_, smallJets);
   //choose leading four AK4 jets and look for high diJet masses 
   //variables -> lead diJet mass, sublead diJet mass
   double tot_HT = 0;
   treeVars["tot_HT"] = 0;
   std::vector<TLorentzVector> leadAK4Jets;
   int nAK4 = 0;
   for(auto iJet = smallJets->begin(); iJet != smallJets->end(); iJet++) 
   {
      if((iJet->pt() > 30.)&&(abs(iJet->eta()) < 2.5)  )treeVars["tot_HT"]+=abs(iJet->pt() );

      if( (iJet->pt()  <30.) || (!(iJet->isPFJet())) || (!isgoodjet(iJet->eta(),iJet->neutralHadronEnergyFraction(), iJet->neutralEmEnergyFraction(),iJet->numberOfDaughters(),iJet->chargedHadronEnergyFraction(),iJet->chargedMultiplicity(),iJet->muonEnergyFraction(),iJet->chargedEmEnergyFraction())) ) continue;
      if(nAK4 < 4) leadAK4Jets.push_back(TLorentzVector(iJet->px(),iJet->py(),iJet->pz(),iJet->energy()));
      nAK4++;

      if( ((iJet->bDiscriminator("pfDeepCSVJetTags:probb") + iJet->bDiscriminator("pfDeepCSVJetTags:probbb") )< 0.4184)||(iJet->pt() < 30.) )continue;
      //loose

      //loose is 0.1241 
   }

   //std::cout << nAK4 << std::endl;
   if(nAK4 <4)return;




//////////////////////////////// AK8 jets//////////////////////////////////////
    int nfatjets = 0;
    double tot_jet_px=0,tot_jet_py=0,tot_jet_pz=0, tot_jet_E=0;
    TLorentzVector allAK8(0,0,0,0);

    for(auto iJet = ak8JetsCollection->begin(); iJet != ak8JetsCollection->end(); iJet++)         ////////Over AK8 Jets
    {

      if(  (iJet->isPFJet()) && (isgoodjet(iJet->eta(),iJet->neutralHadronEnergyFraction(), iJet->neutralEmEnergyFraction(),iJet->numberOfDaughters(),iJet->chargedHadronEnergyFraction(),iJet->chargedMultiplicity(),iJet->muonEnergyFraction(),iJet->chargedEmEnergyFraction(),nfatjets)   ))allAK8+=TLorentzVector(iJet->px(),iJet->py(),iJet->pz(),iJet->energy());

      if((sqrt(pow(iJet->mass(),2)+pow(iJet->pt(),2)) < 300.) || ((!iJet->isPFJet())) || (!isgoodjet(iJet->eta(),iJet->neutralHadronEnergyFraction(), iJet->neutralEmEnergyFraction(),iJet->numberOfDaughters(),iJet->chargedHadronEnergyFraction(),iJet->chargedMultiplicity(),iJet->muonEnergyFraction(),iJet->chargedEmEnergyFraction(),nfatjets)) || (iJet->mass()< 0.)) continue; //userFloat("ak8PFJetsPuppiSoftDropMass")
      for (unsigned int iii=0; iii<iJet->numberOfDaughters();iii++)   ///////get all jet particles
      {
         const reco::Candidate* iJ = iJet->daughter(iii);
         const pat::PackedCandidate* candJetbegin = (pat::PackedCandidate*) iJ;
         double puppiweight = candJetbegin->puppiWeight();
         candsUnboosted.push_back(reco::LeafCandidate(iJet->daughter(iii)->charge(), reco::Candidate::LorentzVector(puppiweight*iJ->px(), puppiweight*iJ->py(), puppiweight*iJ->pz(), puppiweight*iJ->energy())));
         tot_jet_px+=puppiweight*candJetbegin->px();tot_jet_py+=puppiweight*candJetbegin->py();tot_jet_pz+=puppiweight*candJetbegin->pz();tot_jet_E+=puppiweight*candJetbegin->energy();
      }
      nfatjets++;
   }

    treeVars["tot_pt"] = allAK8.Pt();
    if (nfatjets < 3){ return; } 

   double tot_jet_p = sqrt(pow(tot_jet_px,2)+pow(tot_jet_py,2)+pow(tot_jet_pz,2)   ); 

    TVector3 totJetBeta = TVector3(tot_jet_px/tot_jet_E,tot_jet_py/tot_jet_E ,tot_jet_pz/tot_jet_E);

   std::vector<TLorentzVector> genChiZtBoosted;
   std::vector<TLorentzVector> genChiHtBoosted;
   std::vector<TLorentzVector> genChiWbBoosted;


   //boost gen Chis into MPP frame
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
        //std::cout << "More than 2 gen Chis ... you did something wrong " << std::endl;
        return;
   }


    //////////boost all jet particles into MPP frame
    std::vector<fastjet::PseudoJet> candsBoostedPJ;

    for(auto iC = candsUnboosted.begin();iC != candsUnboosted.end(); iC++)
    {
        TLorentzVector iC_(iC->px(),iC->py(),iC->pz(),iC->energy());
        iC_.Boost(-totJetBeta.X(),-totJetBeta.Y(),-totJetBeta.Z());
        candsBoostedPJ.push_back(  fastjet::PseudoJet(iC_.Px(), iC_.Py(), iC_.Pz(), iC_.E()   ) );

    }

    double R0 = 0.8;
    //fastjet::JetDefinition jet_def0(fastjet::antikt_algorithm, R0);
    fastjet::JetDefinition jet_def0(fastjet::cambridge_algorithm, R0);
    fastjet::ClusterSequence cs_jet0(candsBoostedPJ, jet_def0); 
    std::vector<TLorentzVector> candsBoostedTLV;   



   // shedding particles here -> can change this energy threshold, just changed it from 50. 
   std::vector<fastjet::PseudoJet> jetsFJ_jet0 = fastjet::sorted_by_E(cs_jet0.inclusive_jets(10.));
   double MPP_COM2_px = 0,MPP_COM2_py=0,MPP_COM2_pz =0,MPP_COM2_E = 0;
   int counter = 0;
   for (auto iJet=jetsFJ_jet0.begin(); iJet<jetsFJ_jet0.end(); iJet++)                             
   {   
      counter++;
      if (iJet->constituents().size() > 0)
      {
         for(auto iDaughter = iJet->constituents().begin(); iDaughter != iJet->constituents().end(); iDaughter++)
         {
            candsBoosted.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iDaughter->px(), iDaughter->py(), iDaughter->pz(), iDaughter->E())));
            candsBoostedTLV.push_back(TLorentzVector( iDaughter->px(), iDaughter->py(), iDaughter->pz(), iDaughter->E()    ));
         }
      }
      MPP_COM2_px+= iJet->px(); MPP_COM2_py+= iJet->py(); MPP_COM2_pz+=iJet->pz() ; MPP_COM2_E+=iJet->E() ;
   }


    TVector3 MPP_COM2 = TVector3(MPP_COM2_px/MPP_COM2_E,MPP_COM2_py/MPP_COM2_E ,MPP_COM2_pz/MPP_COM2_E);
    std::vector<reco::LeafCandidate> candsBoosted_MPPCOM2;   
    for(auto iPart = candsBoostedTLV.begin(); iPart != candsBoostedTLV.end(); iPart++)
    {
        iPart->Boost(-MPP_COM2.X(), -MPP_COM2.Y(), -MPP_COM2.Z());
        candsBoosted_MPPCOM2.push_back( reco::LeafCandidate(+1, reco::Candidate::LorentzVector( iPart->Px(),iPart->Py(),iPart->Pz(),iPart->E()   ) )       );
    }




    Thrust thrust_(candsBoosted.begin(), candsBoosted.end());                             //thrust axis in COM frame
    math::XYZVector thrustAxis = thrust_.axis();
    TVector3 thrust_vector(thrustAxis.X(),thrustAxis.Y(),thrustAxis.Z());

    //get superjet characterizations


    //positiveLobe = positive superjet classification: 1 = Zt, 2 = Ht, 3 = Wb
   std::string positiveLobe = "";
   std::string negativeLobe = "";

   //double tempCos = -9999;
   if ( genChiWb.size() > 0)
   {
        for(auto iG = genChiWb.begin(); iG != genChiWb.end(); iG++ )
        {
            // need angle between thrust vector
            TVector3 candJet_vec = iG->Vect();
            double cosAngle = cos(candJet_vec.Angle(thrust_vector));
            if (cosAngle > 0)
            {
                if ( positiveLobe == "") 
                {
                    //tempCos = cosAngle;
                    positiveLobe = "Wb";
                }
                else 
                {
                    //std::cout << "Two Chis on one side of SJ ... problem. Cosines are: " << tempCos << "/ " << cosAngle << std::endl;
                    return;
                }
            }
            else if (cosAngle < 0)
            {
                if ( negativeLobe == "") 
                {
                    negativeLobe = "Wb";
                    //tempCos = cosAngle;
                }
                else 
                {
                    //std::cout << "Two Chis on one side of SJ ... problem. Cosines are: " << tempCos << "/ " << cosAngle << std::endl;
                    return;
                }
            }
        }   
   }

//need to change these ones.
   if ( genChiZt.size() > 0)
   {
        for(auto iG = genChiZt.begin(); iG != genChiZt.end(); iG++ )
        {
            // need angle between thrust vector
            TVector3 candJet_vec = iG->Vect();
            double cosAngle = cos(candJet_vec.Angle(thrust_vector));
            if (cosAngle > 0)
            {
                if ( positiveLobe == "") 
                {
                    //tempCos = cosAngle;
                    positiveLobe = "Zt";
                }
                else 
                {
                    //std::cout << "Two Chis on one side of SJ ... problem. Cosines are: " << tempCos << "/ " << cosAngle << std::endl;
                    return;
                }
            }
            else if (cosAngle < 0)
            {
                if ( negativeLobe == "")
                {
                    //tempCos = cosAngle;
                    negativeLobe = "Zt";
                } 
                else 
                {
                    //std::cout << "Two Chis on one side of SJ ... problem. Cosines are: " << tempCos << "/ " << cosAngle << std::endl;
                    return;
                }
            }
        }   
   }
   if ( genChiHt.size() > 0)
   {
        for(auto iG = genChiHt.begin(); iG != genChiHt.end(); iG++ )
        {
            // need angle between thrust vector
            TVector3 candJet_vec = iG->Vect();
            double cosAngle = cos(candJet_vec.Angle(thrust_vector));
            if (cosAngle > 0)
            {
                if ( positiveLobe == "")
                {
                    //tempCos = cosAngle;
                    positiveLobe = "Ht";
                } 
                else 
                {
                    //std::cout << "Two Chis on one side of SJ ... problem. Cosines are: " << tempCos << "/ " << cosAngle << std::endl;
                    return;
                }
            }
            else if (cosAngle < 0)
            {
                if ( negativeLobe == "")
                {
                    //tempCos = cosAngle;
                    negativeLobe = "Ht";
                } 
                else 
                {
                    //std::cout << "Two Chis on one side of SJ ... problem. Cosines are: " << tempCos << "/ " << cosAngle << std::endl;
                    return;
                }
            }
        }   
   }
   //std:;cout << "Designations are pos/neg = " << positiveLobe << "/" << negativeLobe << std::endl;

    /////////////////////////////////////////////////////
    //sort jets in terms of angle relative to thrust axis
    std::vector<TLorentzVector> negSuperJet;
    std::vector<TLorentzVector> posSuperJet;
    std::vector<TLorentzVector> miscJets;
    for (auto iJet=jetsFJ_jet0.begin(); iJet<jetsFJ_jet0.end(); iJet++)                             
    {
        TLorentzVector candJet(iJet->px(),iJet->py(),iJet->pz(),iJet->E());
        TVector3 candJet_vec = candJet.Vect();
        double cosAngle = cos(candJet_vec.Angle(thrust_vector));
        if    (cosAngle < -0.999) negSuperJet.push_back(candJet);  
        else if(cosAngle > 0.999) posSuperJet.push_back(candJet);
        else{ miscJets.push_back(candJet); }
    }


   if(miscJets.size() > 0)
   {   
      sortJets testing(negSuperJet,posSuperJet,miscJets);
      negSuperJet = testing.finalSuperJet1;
      posSuperJet = testing.finalSuperJet2;

   }
    std::vector<fastjet::PseudoJet> superJetOne;        //jets for dot prduct #cos(theta) > 0
    std::vector<fastjet::PseudoJet> superJetTwo;       //jets for dot product #cos(theta) < 0 
    std::vector<TLorentzVector> superJetOneLV;        //workaround for weird fastjet bug
    std::vector<TLorentzVector> superJetTwoLV;        //workaround for weird fastjet bug


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
        if      (isMatchedtoSJ(posSuperJet,candJet)) SJMatch = 1;
        else if (isMatchedtoSJ(negSuperJet,candJet)) SJMatch = 2;
        else if(SJMatch == 0) std::cout << "Something didn't work... " << std::endl;

        //sort jet particles into either SuperJet 1 or SuperJet 2
        if (iJet->constituents().size() > 0)
        {
            for(auto iJ = iJet->constituents().begin(); iJ != iJet->constituents().end(); iJ++)
            {  
                if     (SJMatch==1)superJetOneLV.push_back(TLorentzVector(iJ->px(),iJ->py(),iJ->pz(),iJ->E()));
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



    double superJetpx,superJetpy,superJetpz,superJetE;
    std::vector<std::vector<fastjet::PseudoJet>> superJets;
    superJets.push_back(superJetOne);
    superJets.push_back(superJetTwo);

    //superjet classification: 1 = Zt, 2 = Ht, 3 = Wb
    //these look fine -> move this into superjet COM frame and compare vs the different AK4 jets
    //maybe try looking at ALL particles in SJ COM frame, not just the leading 4 AK4 jets


    std::vector<fastjet::PseudoJet> SJParts1;   
    std::vector<fastjet::PseudoJet> SJParts2; 

    int nSuperJets = 0;
    int SJCounter = 0;
    //std::cout << "SJ1 has " << superJetOne.size() << " particles, and SJ2 has " << superJetTwo.size() << " particles" << std::endl;
    for(auto iSJ = superJets.begin();iSJ!= superJets.end();iSJ++)
    {

        SJCounter++;
        if      (  ((SJCounter == 1) && (positiveLobe != jetType_)) || ((SJCounter == 2) && (negativeLobe != jetType_))    )
        {
            //std::cout << "Some event didn't make it - " << positiveLobe << "/" << negativeLobe << std::endl;
            return;
        }
        if((SJCounter == 1) && (positiveLobe != jetType_)) continue;
        else if((SJCounter == 2) && (negativeLobe != jetType_)) continue;
        
        nPassedEvents++;

        superJetpx=0;superJetpy=0;superJetpz=0;superJetE=0;
        for(auto iP = iSJ->begin();iP != iSJ->end();iP++)
        {
            superJetpx+=iP->px();superJetpy+=iP->py();superJetpz+=iP->pz();superJetE +=iP->E();
        }

        TLorentzVector superJetTLV(superJetpx,superJetpy,superJetpz,superJetE);    //Lorentz vector representing jet axis -> now minimize the parallel momentum
      
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
            sumPz += iP_->pz();
            sumP += abs(sqrt(pow(iP_->pz(),2) + pow(iP_->px(),2)+ pow(iP_->py(),2)));
        }


        ///reclustering SuperJet that is now boosted into the SJ COM frame
        double R = 0.4;

        //fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
        fastjet::JetDefinition jet_def(fastjet::cambridge_algorithm, R);
        fastjet::ClusterSequence cs_jet(boostedSuperJetPart, jet_def); 
        std::vector<fastjet::PseudoJet> jetsFJ_jet = fastjet::sorted_by_E(cs_jet.inclusive_jets(0.0));


        double SJ_25_px = 0, SJ_25_py=0,SJ_25_pz=0,SJ_25_E=0;
        double SJ_50_px = 0, SJ_50_py=0,SJ_50_pz=0,SJ_50_E=0;
        double SJ_100_px = 0, SJ_100_py=0,SJ_100_pz=0,SJ_100_E=0;
        double SJ_150_px = 0, SJ_150_py=0,SJ_150_pz=0,SJ_150_E=0;
        double SJ_200_px = 0, SJ_200_py=0,SJ_200_pz=0,SJ_200_E=0;
        double SJ_300_px = 0, SJ_300_py=0,SJ_300_pz=0,SJ_300_E=0;

        int SJ_nAK4_25_ = 0;
        int SJ_nAK4_50_ = 0,SJ_nAK4_100_ = 0,SJ_nAK4_150_ = 0,SJ_nAK4_200_ = 0,SJ_nAK4_300_ = 0;

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
                    if(nSuperJets == 0)SJParts1.push_back( fastjet::PseudoJet( iPart->px(), iPart->py(),iPart->pz(),iPart->E()    )   );
                    else if(nSuperJets == 0)SJParts2.push_back( fastjet::PseudoJet( iPart->px(), iPart->py(),iPart->pz(),iPart->E()    )   );

                }
            }

            
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
            pseudoJetNum++;
        }

        //std::cout << "SJ COM particles for superjet " << nSuperJets << std::endl;
        //if(nSuperJets == 0)
        //{
          //  for(auto iP = SJParts1.begin(); iP!=SJParts1.end();iP++)
          //  {
            //    std::cout << iP->px() << " " << iP->py() << " " << iP->pz() << " " << iP->E() << std::endl;
           // }
       // }
        //else if(nSuperJets==1)
        //{
         //   for(auto iP = SJParts2.begin(); iP!=SJParts2.end();iP++)
           // {
           //     std::cout << iP->px() << " " << iP->py() << " " << iP->pz() << " " << iP->E() << std::endl;
           // }
       // }
        //std::cout << "There are " << jetsFJ_jet[0].constituents().size() + jetsFJ_jet[1].constituents().size() + jetsFJ_jet[2].constituents().size() + jetsFJ_jet[3].constituents().size() << " total particles in leading 4 reclustered AK4 jets in SJ "<< nSuperJets << std::endl;
        boostedSuperJetPart.clear();   //shouldn't be needed, just in case
        //std::cout << "--------------New Superjet--------------" << std::endl;
        nSuperJets++; 


        /////annoying process of getting the BES information for the reclustered AK4 jets

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
        double sumP_AK41 = 0,sumP_AK42 = 0,sumP_AK43 = 0, sumP_AK44 = 0;
        for(auto iP = AK41_parts.begin(); iP != AK41_parts.end(); iP++)
        {
            iP->Boost(-AK41_boost.X(),-AK41_boost.Y(), -AK41_boost.Z());
            boostedAK41_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
            boostedAK41_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
            boostedAK41_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() ));
            sumPz_AK41+=iP->Pz(); sumP_AK41+= abs(iP->P());
        }
        for(auto iP = AK42_parts.begin(); iP != AK42_parts.end(); iP++)
        {
            iP->Boost(-AK42_boost.X(),-AK42_boost.Y(), -AK42_boost.Z());
            boostedAK42_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
            boostedAK42_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
            boostedAK42_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() ));  
            sumPz_AK42+=iP->Pz(); sumP_AK42+= abs(iP->P());

        }
        for(auto iP = AK43_parts.begin(); iP != AK43_parts.end(); iP++)
        {
            iP->Boost(-AK43_boost.X(),-AK43_boost.Y(), -AK43_boost.Z());
            boostedAK43_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
            boostedAK43_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
            boostedAK43_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() )); 
            sumPz_AK43 += iP->Pz(); sumP_AK43 += abs(iP->P());   
        }
        for(auto iP = AK44_parts.begin(); iP != AK44_parts.end(); iP++)
        {
            iP->Boost(-AK44_boost.X(),-AK44_boost.Y(), -AK44_boost.Z());
            boostedAK44_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
            boostedAK44_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
            boostedAK44_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() )); 
            sumPz_AK44 += iP->Pz(); sumP_AK44 += abs(iP->P());   
        }

        ////vectors to get angles
        
        TVector3 AK4_jet1(jetsFJ_jet[0].px(),jetsFJ_jet[0].py(),jetsFJ_jet[0].pz());
        TVector3 AK4_jet2(jetsFJ_jet[1].px(),jetsFJ_jet[1].py(),jetsFJ_jet[1].pz());
        TVector3 AK4_jet3(jetsFJ_jet[2].px(),jetsFJ_jet[2].py(),jetsFJ_jet[2].pz());
        TVector3 AK4_jet4(jetsFJ_jet[3].px(),jetsFJ_jet[3].py(),jetsFJ_jet[3].pz());

        //fill superjet variables here ...
        //SJ mass variables
        treeVars["SJ_mass"]    = sqrt(pow(superJetE,2)-pow(superJetpx,2)-pow(superJetpy,2)-pow(superJetpz,2)); 
        treeVars["SJ_mass_25"] = sqrt(pow(SJ_25_E,2)-pow(SJ_25_px,2)-pow(SJ_25_py,2)-pow(SJ_25_pz,2)); 
        treeVars["SJ_mass_50"] = sqrt(pow(SJ_50_E,2)-pow(SJ_50_px,2)-pow(SJ_50_py,2)-pow(SJ_50_pz,2)); 
        treeVars["SJ_mass_100"] = sqrt(pow(SJ_100_E,2)-pow(SJ_100_px,2)-pow(SJ_100_py,2)-pow(SJ_100_pz,2)); 
        treeVars["SJ_mass_150"] = sqrt(pow(SJ_150_E,2)-pow(SJ_150_px,2)-pow(SJ_150_py,2)-pow(SJ_150_pz,2)); 
        treeVars["SJ_mass_200"] = sqrt(pow(SJ_200_E,2)-pow(SJ_200_px,2)-pow(SJ_200_py,2)-pow(SJ_200_pz,2)); 
        treeVars["SJ_mass_300"] = sqrt(pow(SJ_300_E,2)-pow(SJ_300_px,2)-pow(SJ_300_py,2)-pow(SJ_300_pz,2)); 


        //SJ nAK4 variables
        treeVars["SJ_nAK4_25"] = SJ_nAK4_25_; 
        treeVars["SJ_nAK4_50"] = SJ_nAK4_50_; 
        treeVars["SJ_nAK4_100"] = SJ_nAK4_100_; 
        treeVars["SJ_nAK4_150"] = SJ_nAK4_150_; 
        treeVars["SJ_nAK4_200"] = SJ_nAK4_200_; 
        treeVars["SJ_nAK4_300"] = SJ_nAK4_300_; 

        //softdrop mass???
        //bjet particle percentages

        //AK4 jet mass combinations
        treeVars["AK4_m1"] = jetsFJ_jet[0].m();   
        treeVars["AK4_m2"] = jetsFJ_jet[1].m(); 
        treeVars["AK4_m3"] = jetsFJ_jet[2].m(); 
        treeVars["AK4_m4"] = jetsFJ_jet[3].m(); 

        treeVars["AK41_E"] = jetsFJ_jet[0].E(); 
        treeVars["AK42_E"] = jetsFJ_jet[1].E(); 
        treeVars["AK43_E"] = jetsFJ_jet[2].E(); 
        treeVars["AK44_E"] = jetsFJ_jet[3].E(); 


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
        treeVars["AK4_theta12"] = abs(AK4_jet1.Angle(AK4_jet2));   
        treeVars["AK4_theta13"] = abs(AK4_jet1.Angle(AK4_jet3));
        treeVars["AK4_theta14"] = abs(AK4_jet1.Angle(AK4_jet4));
        treeVars["AK4_theta23"] = abs(AK4_jet2.Angle(AK4_jet3));
        treeVars["AK4_theta24"] = abs(AK4_jet2.Angle(AK4_jet4));
        treeVars["AK4_theta34"] = abs(AK4_jet3.Angle(AK4_jet4));

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


        //AK4 jet boosted information - boost reclustered AK4 jets into their COM and look at BES variables, ndaughters, nsubjettiness
        treeVars["AK41_ndaughters"] = jetsFJ_jet[0].constituents().size(); 
        //treeVars["AK41_Tau32"] = ; 
        //treeVars["AK41_Tau21"] = ; 
        treeVars["AK41_nsubjets"] = jetsFJ_jet[0].n_exclusive_subjets(0.2); 
        treeVars["AK41_thrust"] = thrustCalculatorAK41.thrust();
        treeVars["AK41_sphericity"] = eventShapesAK41.sphericity();
        treeVars["AK41_asymmetry"] = sumPz_AK41/sumP_AK41; 
        treeVars["AK41_isotropy"] = eventShapesAK41.isotropy();
        treeVars["AK41_aplanarity"] = eventShapesAK41.aplanarity();
        treeVars["AK41_FW1"] = fwmAK41[1]; 
        treeVars["AK41_FW2"] = fwmAK41[2]; 
        treeVars["AK41_FW3"] = fwmAK41[3]; 
        treeVars["AK41_FW4"] = fwmAK41[4]; 

        treeVars["AK42_ndaughters"] = jetsFJ_jet[1].constituents().size(); 
        //treeVars["AK42_Tau32"] = ; 
        //treeVars["AK42_Tau21"] = ;
        treeVars["AK42_nsubjets"] = jetsFJ_jet[1].exclusive_subjets(0.2).size();
        treeVars["AK42_thrust"] = thrustCalculatorAK42.thrust(); 
        treeVars["AK42_sphericity"] = eventShapesAK42.sphericity();
        treeVars["AK42_asymmetry"] = sumPz_AK42/sumP_AK42; 
        treeVars["AK42_isotropy"] = eventShapesAK42.isotropy();
        treeVars["AK42_aplanarity"] = eventShapesAK42.aplanarity();
        treeVars["AK42_FW1"] = fwmAK42[1]; 
        treeVars["AK42_FW2"] = fwmAK42[2]; 
        treeVars["AK42_FW3"] = fwmAK42[3]; 
        treeVars["AK42_FW4"] = fwmAK42[4]; 

        treeVars["AK43_ndaughters"] = jetsFJ_jet[2].constituents().size(); 
        //treeVars["AK43_Tau32"] = ; 
        //treeVars["AK43_Tau21"] = ; 
        treeVars["AK43_nsubjets"] = jetsFJ_jet[2].exclusive_subjets(0.2).size();
        treeVars["AK43_thrust"] = thrustCalculatorAK43.thrust();
        treeVars["AK43_sphericity"] = eventShapesAK43.sphericity();
        treeVars["AK43_asymmetry"] = sumPz_AK43/sumP_AK43; 
        treeVars["AK43_isotropy"] = eventShapesAK43.isotropy();
        treeVars["AK43_aplanarity"] = eventShapesAK43.aplanarity();
        treeVars["AK43_FW1"] = fwmAK43[1]; 
        treeVars["AK43_FW2"] = fwmAK43[2]; 
        treeVars["AK43_FW3"] = fwmAK43[3]; 
        treeVars["AK43_FW4"] = fwmAK43[4]; 

        //treeVars["AK44_ndaughters"] = jetsFJ_jet[3].constituents().size(); 
        //treeVars["AK43_Tau32"] = ; 
        //treeVars["AK43_Tau21"] = ; 
        //treeVars["AK44_nsubjets"] = jetsFJ_jet[3].exclusive_subjets(0.2).size();
        //treeVars["AK44_thrust"] = thrustCalculatorAK44.thrust();
        //treeVars["AK44_sphericity"] = eventShapesAK44.sphericity();
        //treeVars["AK44_asymmetry"] = sumPz_AK44/sumP_AK44; 
        //treeVars["AK44_isotropy"] = eventShapesAK44.isotropy();
        //treeVars["AK44_aplanarity"] = eventShapesAK44.aplanarity();
        //treeVars["AK44_FW1"] = fwmAK44[1]; 
        //treeVars["AK44_FW2"] = fwmAK44[2]; 
        //treeVars["AK44_FW3"] = fwmAK44[3]; 
        //treeVars["AK44_FW4"] = fwmAK44[4]; 

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
        for (unsigned i = 0; i < listOfVars.size(); i++)
        {
            if (  (treeVars[ listOfVars[i] ] != treeVars[ listOfVars[i] ] ) || ( isinf(treeVars[ listOfVars[i] ])   )  )
            {
                //std::cout << "Skipped Event " << std::endl;
                return;
            }
            else if ( abs(treeVars[ listOfVars[i] ]+999.99 ) < 1.0e-10 ) return;
        }
        jetTree->Fill();
        for (unsigned i = 0; i < listOfVars.size(); i++)
        {
            if ( (listOfVars[i] ==  "tot_HT" ) || (listOfVars[i] == "tot_pt") ) continue;
            treeVars[ listOfVars[i] ] = -999.99;
        }
    }



}


//========================================t=========================================
// Method called once each job just before starting event loop  -------------------
//=================================================================================

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

//=================================================================================
// Method fills 'descriptions' with the allowed parameters for the module  --------
//=================================================================================

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






