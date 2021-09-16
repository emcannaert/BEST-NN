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
     This EDProducer is meant to be used with CMSSW_10_6_27                            ---
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

    // enumerate possible jet types
    enum JetType { Q, H, t, W, Z, b};

    // enumerate possible jet collections
    enum JetColl{ CHS, PUPPI};

    // create a struct to help with mapping string label to enum value
    struct JetTypeStringToEnum {
        const char label;
        JetType value;
    };

    // create a struct to help with mapping string label to enum value
    struct JetCollStringToEnum {
        const char* label;
        JetColl value;
    };

    // Create a mapping from the input jetType to enum
    JetType jetTypeFromString(const std::string& label) {
        static const JetTypeStringToEnum jetTypeStringToEnumMap[] = {
            {'Q', Q},
            {'H', H},
            {'t', t},
            {'W', W},
            {'Z', Z},
            {'b', b}
        };

        JetType value = (JetType)-1;
        bool found = false;
        for (int i = 0; jetTypeStringToEnumMap[i].label && (!found); ++i){
            if (!strcmp(label.c_str(), &jetTypeStringToEnumMap[i].label) ) {
                found = true;
                value = jetTypeStringToEnumMap[i].value;
            }
        }

        // Throw an error if user inputs an unrecognized type
        if (!found){
            throw cms::Exception("JetTypeError") << label << " is not a recognized JetType";
        }

        return value;
    }


    // Create a mapping from the input jetColl to enum
    JetColl jetCollFromString(const std::string& label) {
        static const JetCollStringToEnum jetCollStringToEnumMap[] = {
            {"CHS", CHS},
            {"PUPPI", PUPPI}
        };

        JetColl value = (JetColl)-1;
        bool found = false;
        for (int i = 0; jetCollStringToEnumMap[i].label && (!found); ++i){
            if (!strcmp(label.c_str(), jetCollStringToEnumMap[i].label) ) {
                found = true;
                value = jetCollStringToEnumMap[i].value;
            }
        }

        // Throw an error if user inputs an unrecognized type
        if (!found){
            throw cms::Exception("JetCollError") << label << " is not a recognized JetColl";
        }

        return value;
    }
}


///////////////////////////////////////////////////////////////////////////////////
// Class declaration --------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

class BESTProducer : public edm::stream::EDProducer<> {
   public:
      explicit BESTProducer(const edm::ParameterSet&);
      ~BESTProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

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
      std::string inputJetColl_;
      best::JetType jetType_;
      best::JetColl jetColl_;
      bool storeDaughters;
      bool storeJetImages;

      // Tree variables
      TTree *jetTree;
      std::map<std::string, float> treeVars;
      std::vector<std::string> listOfVars;
      std::map<std::string, std::vector<float> > jetVecVars;
      std::vector<std::string> listOfVecVars;
      std::map<std::string, std::array<std::array<std::array<float, 16>, 31>, 31> > imgVars;
      std::vector<std::string> listOfImgVars;

      // Tokens
      //edm::EDGetTokenT<std::vector<pat::PackedCandidate> > pfCandsToken_;
      edm::EDGetTokenT<std::vector<pat::Jet> > ak8JetsToken_;
      //edm::EDGetTokenT<std::vector<pat::Jet> > ak4JetsToken_;
      edm::EDGetTokenT<std::vector<reco::GenParticle> > genPartToken_;
      edm::EDGetTokenT<std::vector<reco::VertexCompositePtrCandidate> > secVerticesToken_;
      edm::EDGetTokenT<std::vector<reco::Vertex> > verticesToken_;

      //edm::EDGetTokenT<std::vector<pat::Jet> > ak8CHSSoftDropSubjetsToken_;

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

BESTProducer::BESTProducer(const edm::ParameterSet& iConfig):
    inputJetColl_ (iConfig.getParameter<std::string>("inputJetColl")),
    jetType_ (best::jetTypeFromString(iConfig.getParameter<std::string>("jetType"))),
    jetColl_ (best::jetCollFromString(iConfig.getParameter<std::string>("jetColl"))),
    storeDaughters (iConfig.getParameter<bool>("storeDaughters")),
    storeJetImages (iConfig.getParameter<bool>("storeJetImages"))
{

    std::cout<<"Jet Colls, "<<inputJetColl_<<", "<<jetColl_<<std::endl;
    //------------------------------------------------------------------------------
    // Prepare TFile Service -------------------------------------------------------
    //------------------------------------------------------------------------------

    edm::Service<TFileService> fs;
    jetTree = fs->make<TTree>("jetTree","jetTree");

    //------------------------------------------------------------------------------
    // Create tree variables and branches ------------------------------------------
    //------------------------------------------------------------------------------
    // listOfVars is the flat part of the TTree ------------------------------------
    // listOfVecVars is the vector part of the TTree -------------------------------
    //------------------------------------------------------------------------------

    // AK8 jet variables
    listOfVars.push_back("nJets");

    listOfVars.push_back("jetAK8_phi");
    listOfVars.push_back("jetAK8_eta");
    listOfVars.push_back("jetAK8_pt");
    listOfVars.push_back("jetAK8_mass");
    listOfVars.push_back("jetAK8_energy");
    listOfVars.push_back("jetAK8_SoftDropMass");
    listOfVars.push_back("jetAK8_charge");

    // Deep AK8
    listOfVars.push_back("jetAK8_deepAK8_rawL");
    listOfVars.push_back("jetAK8_deepAK8_rawC");
    listOfVars.push_back("jetAK8_deepAK8_rawB");
    listOfVars.push_back("jetAK8_deepAK8_rawW");
    listOfVars.push_back("jetAK8_deepAK8_rawZ");
    listOfVars.push_back("jetAK8_deepAK8_rawH");
    listOfVars.push_back("jetAK8_deepAK8_rawT");
    listOfVars.push_back("jetAK8_deepAK8_dnn_Largest");
    listOfVars.push_back("jetAK8_deepAK8MD_rawL");
    listOfVars.push_back("jetAK8_deepAK8MD_rawC");
    listOfVars.push_back("jetAK8_deepAK8MD_rawB");
    listOfVars.push_back("jetAK8_deepAK8MD_rawW");
    listOfVars.push_back("jetAK8_deepAK8MD_rawZ");
    listOfVars.push_back("jetAK8_deepAK8MD_rawH");
    listOfVars.push_back("jetAK8_deepAK8MD_rawT");
    listOfVars.push_back("jetAK8_deepAK8MD_dnn_Largest");

    // b-tagging
    listOfVars.push_back("jetAK8_bDisc_pfDeepCSVJetTags_probb");
    listOfVars.push_back("jetAK8_bDisc_pfDeepCSVJetTags_probbb");
    listOfVars.push_back("jetAK8_bDisc_pfCombinedInclusiveSecondaryVertexV2BJetTags");
    listOfVars.push_back("jetAK8_bDisc_pfBoostedDoubleSecondaryVertexAK8BJetTags");

    // Vertex Variables
    listOfVars.push_back("nSecondaryVertices");
    listOfVecVars.push_back("SV_pt"); // Possible bug!
    listOfVecVars.push_back("SV_eta");
    listOfVecVars.push_back("SV_phi");
    listOfVecVars.push_back("SV_mass");
    listOfVecVars.push_back("SV_nTracks");
    listOfVecVars.push_back("SV_chi2");
    listOfVecVars.push_back("SV_Ndof");

    // Deep Jet b Discriminants
    listOfVars.push_back("bDisc");
    listOfVars.push_back("bDisc1");
    listOfVars.push_back("bDisc2");

    // nsubjettiness
    listOfVars.push_back("jetAK8_Tau4");
    listOfVars.push_back("jetAK8_Tau3");
    listOfVars.push_back("jetAK8_Tau2");
    listOfVars.push_back("jetAK8_Tau1");
    listOfVars.push_back("jetAK8_Tau32");
    listOfVars.push_back("jetAK8_Tau21");

    // Define vector of rest masses (in GeV) to boost to (rather than the individual H, t, W, Z masses).
    std::vector<int> restMasses;
    restMasses.clear();
    unsigned int iterMass = 1;
    while(iterMass <= 200) {
        restMasses.push_back(iterMass); // Add this mass to the vector, then increment by 1 GeV if any condition is true, or 5 GeV if none are true. 
        iterMass += ( (iterMass < 15) || (iterMass >= 80 && iterMass < 95) || (iterMass >= 165 && iterMass < 180) ) ? 1: 5;
    }
    // Now use this vector to generate the variable names to add:
    for (unsigned int imass=0; imass < restMasses.size(); imass++) {
        std::string frame = std::to_string(restMasses[imass])+"GeV";

        // Fox Wolfram Moments
        listOfVars.push_back("FoxWolfH1_"+frame);
        listOfVars.push_back("FoxWolfH2_"+frame);
        listOfVars.push_back("FoxWolfH3_"+frame);
        listOfVars.push_back("FoxWolfH4_"+frame);

        // Event Shape Variables
        listOfVars.push_back("isotropy_"+frame);
        listOfVars.push_back("sphericity_"+frame);
        listOfVars.push_back("aplanarity_"+frame);
        listOfVars.push_back("thrust_"+frame);

        // Jet Mass
        listOfVars.push_back("nJets_"+frame);

        listOfVars.push_back("jet12_mass_"+frame);
        listOfVars.push_back("jet23_mass_"+frame);
        listOfVars.push_back("jet13_mass_"+frame);
        listOfVars.push_back("jet1234_mass_"+frame);

        //Subjet CosTheta and delta CosTheta
        listOfVars.push_back("jet12_CosTheta_"+frame);
        listOfVars.push_back("jet23_CosTheta_"+frame);
        listOfVars.push_back("jet13_CosTheta_"+frame);
        listOfVars.push_back("jet1234_CosTheta_"+frame);

        listOfVars.push_back("jet12_DeltaCosTheta_"+frame);
        listOfVars.push_back("jet13_DeltaCosTheta_"+frame);
        listOfVars.push_back("jet23_DeltaCosTheta_"+frame);

        // Jet Asymmetry
        listOfVars.push_back("asymmetry_"+frame);
    
        // add the daughter and rest frame information
        if(storeDaughters == true){

            // Jet PF Candidate Variables
            listOfVecVars.push_back(frame+"Frame_PF_candidate_px");
            listOfVecVars.push_back(frame+"Frame_PF_candidate_py");
            listOfVecVars.push_back(frame+"Frame_PF_candidate_pz");
            listOfVecVars.push_back(frame+"Frame_PF_candidate_energy");

            // rest frame subjet variables
            listOfVecVars.push_back(frame+"Frame_jet_px");
            listOfVecVars.push_back(frame+"Frame_jet_py");
            listOfVecVars.push_back(frame+"Frame_jet_pz");
            listOfVecVars.push_back(frame+"Frame_jet_energy");
        }

        // rest frame jet image variables
        if(storeJetImages == true) listOfImgVars.push_back(frame+"Frame_image");

    }     
    restMasses.clear();

    // add the daughter and rest frame information
    if(storeDaughters == true){

        // Jet PF Candidate Variables
        listOfVecVars.push_back("LabFrame_PF_candidate_px");
        listOfVecVars.push_back("LabFrame_PF_candidate_py");
        listOfVecVars.push_back("LabFrame_PF_candidate_pz");
        listOfVecVars.push_back("LabFrame_PF_candidate_energy");
        listOfVecVars.push_back("LabFrame_PF_candidate_mass");
        listOfVecVars.push_back("LabFrame_PF_candidate_ecalEnergy");

        listOfVecVars.push_back("LabFrame_PF_candidate_charge");
        listOfVecVars.push_back("LabFrame_PF_candidate_pdgId");
        listOfVecVars.push_back("LabFrame_PF_candidate_abspdgId");
        listOfVecVars.push_back("LabFrame_PF_candidate_isElectron");
        listOfVecVars.push_back("LabFrame_PF_candidate_isMuon");
        listOfVecVars.push_back("LabFrame_PF_candidate_isPhoton");
        listOfVecVars.push_back("LabFrame_PF_candidate_isNeutralHadron");
        listOfVecVars.push_back("LabFrame_PF_candidate_isChargedHadron");
        
        listOfVecVars.push_back("LabFrame_PF_candidate_deltaEta");
        listOfVecVars.push_back("LabFrame_PF_candidate_deltaPhi");
        listOfVecVars.push_back("LabFrame_PF_candidate_deltaR");
        listOfVecVars.push_back("LabFrame_PF_candidate_logpT");
        listOfVecVars.push_back("LabFrame_PF_candidate_logEnergy");
        listOfVecVars.push_back("LabFrame_PF_candidate_logpTRatio");
        listOfVecVars.push_back("LabFrame_PF_candidate_logEnergyRatio");

        // PUPPI weights
        listOfVecVars.push_back("PUPPI_Weights");

    }

    // Make Branches for each variable
    for (unsigned i = 0; i < listOfVars.size(); i++){
        treeVars[ listOfVars[i] ] = -999.99;
        jetTree->Branch( (listOfVars[i]).c_str() , &(treeVars[ listOfVars[i] ]), (listOfVars[i]+"/F").c_str() );
    }

    // Make Branches for each of the jet constituents' variables
    for (unsigned i = 0; i < listOfVecVars.size(); i++){
        jetTree->Branch( (listOfVecVars[i]).c_str() , &(jetVecVars[ listOfVecVars[i] ]) ); //Possible bug!
    }

    // Make branches for each of the images
    if(storeJetImages == true){
        for (unsigned i = 0; i < listOfImgVars.size(); i++){
            jetTree->Branch( (listOfImgVars[i]).c_str() , &(imgVars[ listOfImgVars[i] ]), (listOfImgVars[i]+"[31][31][16]/F").c_str() );
        }
    }

    //------------------------------------------------------------------------------
    // Define input tags -----------------------------------------------------------
    //------------------------------------------------------------------------------

    // AK8 Jets
    std::cout<<"AK8 Jets input tag: "<<inputJetColl_<<std::endl;
    edm::InputTag ak8JetsTag_;
    //ak8JetsTag_ = edm::InputTag("selectedUpdatedPatJetsAK8WithDeepTags", "", "PAT");
    //ak8JetsTag_ = edm::InputTag("slimmedJetsAK8", "", "PAT");
    ak8JetsTag_ = edm::InputTag(inputJetColl_, "", "PAT");
    //ak8JetsTag_ = edm::InputTag(inputJetColl_, "", "run"); // this may be needed as an option for 2016 mc
    std::cout<<"Before consumes"<<std::endl;
    ak8JetsToken_ = consumes<std::vector<pat::Jet> >(ak8JetsTag_);
    std::cout<<"After consumes"<<std::endl;

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

    std::cout<<"Done??"<<std::endl;
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

void
BESTProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    using namespace fastjet;
    using namespace std;

    typedef reco::Candidate::PolarLorentzVector fourv;

    //------------------------------------------------------------------------------
    // Create miniAOD object collections -------------------------------------------
    //------------------------------------------------------------------------------

    std::cout<<"Start"<<std::endl;
    
    // Find objects corresponding to the token and link to the handle
    Handle< std::vector<pat::Jet> > ak8JetsCollection;
    std::cout<<"Get by Token"<<std::endl;
    iEvent.getByToken(ak8JetsToken_, ak8JetsCollection);
    //std::cout<<ak8JetsCollection<<std::endl;
    std::cout<<"Product"<<std::endl;
    vector<pat::Jet> ak8Jets = *ak8JetsCollection.product();
    std::cout<<"Got AK8 collection"<<std::endl;

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
    // This makes a TLorentz Vector for each generator Heavy Object to use for jet
    // matching
    //------------------------------------------------------------------------------
    // Please note that the Jet Type has been enumerated:
    // QCD -> 0, H -> 1, t -> 2, W -> 3, Z -> 4, b -> 5
    //------------------------------------------------------------------------------
    int pdgID = -99;
    switch(jetType_){
    case 1:
      pdgID = 25;
      break;
    case 2:
      pdgID = 6;
      break;
    case 3:
      pdgID =  24;
      break;
    case 4:
      pdgID = 23;
      break;
    case 5:
      pdgID = 5;
      break;
    default:
      pdgID = -99;
      break;
    }
    // Store heavy particle for jet matching
    std::vector<TLorentzVector> genParticleToMatch;
    if(jetType_ != 0){
        for (vector<reco::GenParticle>::const_iterator genBegin = genPart.begin(), genEnd = genPart.end(), ipart = genBegin; ipart != genEnd; ++ipart){
            if(abs(ipart->pdgId() ) == pdgID){
                genParticleToMatch.push_back( TLorentzVector(ipart->px(), ipart->py(), ipart->pz(), ipart->energy() ) );
            }
        }
    }

    //------------------------------------------------------------------------------
    // AK8 Jet Loop ----------------------------------------------------------------
    //------------------------------------------------------------------------------
    // This loop makes a tree entry for each jet of interest -----------------------
    //------------------------------------------------------------------------------

    // Create structures for storing daughters and rest frame jets
    vector<reco::Candidate * > daughtersOfJet;
    map<string, vector<TLorentzVector> > boostedDaughters;
    map<string, vector<fastjet::PseudoJet> > restJets;

    // Define vector of rest masses (in GeV) to boost to (rather than the individual H, t, W, Z masses).
    std::vector<int> restMasses;
    restMasses.clear();
    unsigned int iterMass = 1;
    while(iterMass <= 200) {
        restMasses.push_back(iterMass); // Add this mass to the vector, then increment by 1 GeV if any condition is true, or 5 GeV if none are true. 
        iterMass += ( (iterMass < 15) || (iterMass >= 80 && iterMass < 95) || (iterMass >= 165 && iterMass < 180) ) ? 1: 5;
    }

    for (vector<pat::Jet>::const_iterator jetBegin = ak8Jets.begin(), jetEnd = ak8Jets.end(), ijet = jetBegin; ijet != jetEnd; ++ijet){
        bool GenMatching = false;
        daughtersOfJet.clear();
        boostedDaughters.clear();
        restJets.clear();
        TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );

        if(ijet->subjets("SoftDropPuppi").size() >=2 && ijet->numberOfDaughters() > 2 && ijet->pt() >= 500 && fabs(ijet->eta()) < 2.4 &&ijet->userFloat("ak8PFJetsPuppiSoftDropMass") > 10) {

            // gen particle loop, only relevant for non-QCD jets
            if (jetType_ !=0){
                for (size_t iGenParticle = 0; iGenParticle < genParticleToMatch.size(); iGenParticle++){
                    // Check if jet matches any saved genParticle
                    if(jet.DeltaR(genParticleToMatch[iGenParticle]) < 0.1){
                                GenMatching = true;
                    }
                }
            }
            if (GenMatching || (jetType_ == 0)){

                // Store Jet Variables
                treeVars["nJets"] = ak8Jets.size();
                storeJetVariables(treeVars, ijet, jetColl_);

                // Secondary Vertex Variables
                storeSecVertexVariables(treeVars, jetVecVars, jet, secVertices);

                // Get all of the Jet's daughters
                getJetDaughters(daughtersOfJet, ijet);
                if (daughtersOfJet.size() < 3) continue;

                for (unsigned int imass=0; imass < restMasses.size(); imass++) {
                    // Calculate all rest frame variables, skip both loops if frame doesn't work
                    if (calcBESvariables(treeVars, daughtersOfJet, boostedDaughters, ijet, restJets, imgVars, restMasses[imass], storeJetImages ) == false) goto endjetloop;
                } 
                
                // store daughters, rest frame daughters, and rest frame jets
                if(storeDaughters == true){
                    storeJetDaughters(daughtersOfJet, ijet, boostedDaughters, restJets, restMasses, jetVecVars, jetColl_ );
                }

                // Fill the jet entry tree
                jetTree->Fill();
                endjetloop:; // When goto is triggered in the imass loop, the code jumps to here. This is like using "continue" twice, to skip this iteration of the ijet loop.
            }
        }

        //-------------------------------------------------------------------------------
        // Clear and Reset all tree variables -------------------------------------------
        //-------------------------------------------------------------------------------
        for (unsigned i = 0; i < listOfVars.size(); i++){
            treeVars[ listOfVars[i] ] = -999.99;
        }
        for (unsigned i = 0; i < listOfVecVars.size(); i++){
            jetVecVars[ listOfVecVars[i] ].clear();
        }
        /*
        for (unsigned i = 0; i < listOfImgVars.size(); i++){
            for (unsigned j = 0; j < imgVars[ listOfImgVars[i] ].size(); j++) {
                for (unsigned k = 0; k < imgVars[ listOfImgVars[i] ][j].size(); k++) {
                    imgVars[ listOfImgVars[i] ][j][k].clear();
            }
            }
        }
        */
    }

    // Delete vector
    daughtersOfJet.clear();
    boostedDaughters.clear();
    restJets.clear();
    restMasses.clear();
}


//=================================================================================
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
