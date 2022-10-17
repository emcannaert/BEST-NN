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
// Authors: Brendan Regnery, Justin Pilot, Reyer Band, Devin Taylor, Sam Abbott ----------
//         Created:  WED, 8 Aug 2018 21:00:28 GMT  ---------------------------------------
//========================================================================================
//////////////////////////////////////////////////////////////////////////////////////////

// This version of BEST includes the different subjet matching methods we explored while testing

// system include files
#include <memory>
#include <thread>
#include <iostream>
#include <set>

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
        std::string inputSubJetColl_;
        best::JetType jetType_;
        best::JetColl jetColl_;
        bool storeDaughters;

        // Tree variables
        TTree *jetTree;
        std::map<std::string, float> treeVars;
        std::vector<std::string> listOfVars;
        std::map<std::string, std::vector<float> > jetVecVars;
        std::vector<std::string> listOfVecVars;

        // List of strings of the rest masses for each frame that BEST boosts to
        std::vector<std::string> restMasses; 

        // Tokens
        //edm::EDGetTokenT<std::vector<pat::PackedCandidate> > pfCandsToken_;
        edm::EDGetTokenT<std::vector<pat::Jet> > ak8JetsToken_;
        edm::EDGetTokenT<std::vector<pat::Jet> > subJetsToken_;
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
    inputSubJetColl_ (iConfig.getParameter<std::string>("inputSubJetColl")),
    jetType_ (best::jetTypeFromString(iConfig.getParameter<std::string>("jetType"))),
    jetColl_ (best::jetCollFromString(iConfig.getParameter<std::string>("jetColl"))),
    storeDaughters (iConfig.getParameter<bool>("storeDaughters")) {

    //------------------------------------------------------------------------------
    // Prepare Rest Masses ---------------------------------------------------------
    //------------------------------------------------------------------------------

    // Define vector of rest masses (in GeV) to boost to (rather than the individual H, t, W, Z masses).
    restMasses.push_back("300GeV");
    restMasses.push_back("400GeV");
    restMasses.push_back("W");
    restMasses.push_back("Higgs");
    restMasses.push_back("Top");
    restMasses.push_back("ak8");
    restMasses.push_back("ak8SoftDrop");

    //------------------------------------------------------------------------------
    // Prepare TFile Service -------------------------------------------------------
    //------------------------------------------------------------------------------

    // Create the root TTree
    edm::Service<TFileService> fs;
    jetTree = fs->make<TTree>("jetTree","jetTree");

    //------------------------------------------------------------------------------
    // Create tree variables and branches ------------------------------------------
    //------------------------------------------------------------------------------
    // listOfVars is the flat part of the TTree ------------------------------------
    // listOfVecVars is the vector part of the TTree -------------------------------
    //------------------------------------------------------------------------------

    // AK8 jet variables
    // listOfVars.push_back("nJets");

    listOfVars.push_back("jetAK8_phi");
    listOfVars.push_back("jetAK8_eta");
    listOfVars.push_back("jetAK8_pt");
    listOfVars.push_back("jetAK8_mass");
    listOfVars.push_back("jetAK8_SoftDropMass");
    listOfVars.push_back("jetAK8_charge");

    // Deep AK8
    // listOfVars.push_back("jetAK8_deepAK8_rawL");
    // listOfVars.push_back("jetAK8_deepAK8_rawC");
    // listOfVars.push_back("jetAK8_deepAK8_rawB");
    // listOfVars.push_back("jetAK8_deepAK8_rawW");
    // listOfVars.push_back("jetAK8_deepAK8_rawZ");
    // listOfVars.push_back("jetAK8_deepAK8_rawH");
    // listOfVars.push_back("jetAK8_deepAK8_rawT");
    // listOfVars.push_back("jetAK8_deepAK8_dnn_Largest");
    // listOfVars.push_back("jetAK8_deepAK8MD_rawL");
    // listOfVars.push_back("jetAK8_deepAK8MD_rawC");
    // listOfVars.push_back("jetAK8_deepAK8MD_rawB");
    // listOfVars.push_back("jetAK8_deepAK8MD_rawW");
    // listOfVars.push_back("jetAK8_deepAK8MD_rawZ");
    // listOfVars.push_back("jetAK8_deepAK8MD_rawH");
    // listOfVars.push_back("jetAK8_deepAK8MD_rawT");
    // listOfVars.push_back("jetAK8_deepAK8MD_dnn_Largest");

    listOfVars.push_back("jetAK8_deepAK8_probQCDothers");
    listOfVars.push_back("jetAK8_deepAK8_probQCDcc");
    listOfVars.push_back("jetAK8_deepAK8_probQCDbb");
    listOfVars.push_back("jetAK8_deepAK8_probWcq");
    listOfVars.push_back("jetAK8_deepAK8_probHbb"); 
    listOfVars.push_back("jetAK8_deepAK8_probQCDc");
    listOfVars.push_back("jetAK8_deepAK8_probQCDb");
    listOfVars.push_back("jetAK8_deepAK8_probWqq");
    listOfVars.push_back("jetAK8_deepAK8_probZcc");
    listOfVars.push_back("jetAK8_deepAK8_probHcc");
    listOfVars.push_back("jetAK8_deepAK8_probTbqq");
    listOfVars.push_back("jetAK8_deepAK8_probZbb");
    listOfVars.push_back("jetAK8_deepAK8_probZqq");
    listOfVars.push_back("jetAK8_deepAK8_probHqqqq");
    listOfVars.push_back("jetAK8_deepAK8_probTbcq");

    listOfVars.push_back("jetAK8_deepAK8MD_probQCDothers");
    listOfVars.push_back("jetAK8_deepAK8MD_probQCDcc");
    listOfVars.push_back("jetAK8_deepAK8MD_probQCDbb");
    listOfVars.push_back("jetAK8_deepAK8MD_probWcq");
    listOfVars.push_back("jetAK8_deepAK8MD_probHbb"); 
    listOfVars.push_back("jetAK8_deepAK8MD_probQCDc");
    listOfVars.push_back("jetAK8_deepAK8MD_probQCDb");
    listOfVars.push_back("jetAK8_deepAK8MD_probWqq");
    listOfVars.push_back("jetAK8_deepAK8MD_probZcc");
    listOfVars.push_back("jetAK8_deepAK8MD_probHcc");
    listOfVars.push_back("jetAK8_deepAK8MD_probTbqq");
    listOfVars.push_back("jetAK8_deepAK8MD_probZbb");
    listOfVars.push_back("jetAK8_deepAK8MD_probZqq");
    listOfVars.push_back("jetAK8_deepAK8MD_probHqqqq");
    listOfVars.push_back("jetAK8_deepAK8MD_probTbcq");

    // Particle Net
    // listOfVars.push_back("jetAK8_ParticleNet_rawL");
    // listOfVars.push_back("jetAK8_ParticleNet_rawC");
    // listOfVars.push_back("jetAK8_ParticleNet_rawB");
    // listOfVars.push_back("jetAK8_ParticleNet_rawW");
    // listOfVars.push_back("jetAK8_ParticleNet_rawZ");
    // listOfVars.push_back("jetAK8_ParticleNet_rawH");
    // listOfVars.push_back("jetAK8_ParticleNet_rawT");
    // listOfVars.push_back("jetAK8_ParticleNet_dnn_Largest");

    listOfVars.push_back("jetAK8_ParticleNet_probQCDothers");
    listOfVars.push_back("jetAK8_ParticleNet_probQCDcc");
    listOfVars.push_back("jetAK8_ParticleNet_probQCDbb");
    listOfVars.push_back("jetAK8_ParticleNet_probWcq");
    listOfVars.push_back("jetAK8_ParticleNet_probHbb"); 
    listOfVars.push_back("jetAK8_ParticleNet_probQCDc");
    listOfVars.push_back("jetAK8_ParticleNet_probQCDb");
    listOfVars.push_back("jetAK8_ParticleNet_probWqq");
    listOfVars.push_back("jetAK8_ParticleNet_probZcc");
    listOfVars.push_back("jetAK8_ParticleNet_probHcc");
    listOfVars.push_back("jetAK8_ParticleNet_probTbqq");
    listOfVars.push_back("jetAK8_ParticleNet_probZbb");
    listOfVars.push_back("jetAK8_ParticleNet_probZqq");
    listOfVars.push_back("jetAK8_ParticleNet_probHqqqq");
    listOfVars.push_back("jetAK8_ParticleNet_probTbcq");
    listOfVars.push_back("jetAK8_ParticleNet_probTbc");
    listOfVars.push_back("jetAK8_ParticleNet_probTbq");
    
    // Vertex Variables
    listOfVars.push_back("nSecondaryVertices");
    // listOfVecVars.push_back("SV_pt"); // Possible bug!
    // listOfVecVars.push_back("SV_eta");
    // listOfVecVars.push_back("SV_phi");
    // listOfVecVars.push_back("SV_mass");
    // listOfVecVars.push_back("SV_nTracks");
    // listOfVecVars.push_back("SV_chi2");
    // listOfVecVars.push_back("SV_Ndof");

    // Deep Jet b Discriminants
    listOfVars.push_back("bDisc1");
    listOfVars.push_back("bDisc1_probb");
    listOfVars.push_back("bDisc1_probbb");
    listOfVars.push_back("bDisc2");
    listOfVars.push_back("bDisc2_probb");
    listOfVars.push_back("bDisc2_probbb");
    // listOfVars.push_back("bDiscSubJet_Max");
    // listOfVars.push_back("bDiscSubJet_Max_index"); // indexes from 0

    // nsubjettiness
    listOfVars.push_back("jetAK8_Tau4");
    listOfVars.push_back("jetAK8_Tau3");
    listOfVars.push_back("jetAK8_Tau2");
    listOfVars.push_back("jetAK8_Tau1");
    listOfVars.push_back("jetAK8_Tau32");
    listOfVars.push_back("jetAK8_Tau21");

    // Now use this vector to generate the variable names to add:
    for (unsigned int imass=0; imass < restMasses.size(); imass++) {
        // std::string frame = std::to_string(restMasses[imass])+"GeV";
        std::string frame = restMasses[imass];
        if (frame != "Lab"){ // Variables not saved in lab frame:

            // rest frame subjet variables
            listOfVecVars.push_back("jet_px_"+frame);
            listOfVecVars.push_back("jet_py_"+frame);
            listOfVecVars.push_back("jet_pz_"+frame);
            listOfVecVars.push_back("jet_energy_"+frame);

            // Fox Wolfram Moments
            listOfVars.push_back("FoxWolfH1_"+frame);
            listOfVars.push_back("FoxWolfH2_"+frame);
            listOfVars.push_back("FoxWolfH3_"+frame);
            listOfVars.push_back("FoxWolfH4_"+frame);

            // Event Shape Variables
            if (frame == "Higgs") listOfVars.push_back("isotropy");
            listOfVars.push_back("sphericity_"+frame);
            listOfVars.push_back("aplanarity_"+frame);
            listOfVars.push_back("thrust_"+frame);

            // Jet Mass
            if (frame == "Higgs") listOfVars.push_back("nReclusteredJets");

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
        }


        // add the daughter and rest frame information
        // if(storeDaughters == true){

            // Jet PF Candidate Variables
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_px");
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_py");
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_pz");
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_energy");

            // listOfVecVars.push_back(frame+"Frame_PF_candidate_deltaEta");
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_deltaPhi");
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_deltaR");
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_logpT");
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_logEnergy");
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_logpTRatio");
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_logEnergyRatio");

            // // PUPPI weights
            // listOfVecVars.push_back(frame+"Frame_PF_candidate_PUPPI_Weights");

            // if (frame == "Lab"){ // Variables saved only in lab frame:
            //     listOfVecVars.push_back("AllFrame_PF_candidate_charge");
            //     listOfVecVars.push_back("AllFrame_PF_candidate_pdgId");
            //     listOfVecVars.push_back("AllFrame_PF_candidate_abspdgId");
            //     listOfVecVars.push_back("AllFrame_PF_candidate_isElectron");
            //     listOfVecVars.push_back("AllFrame_PF_candidate_isMuon");
            //     listOfVecVars.push_back("AllFrame_PF_candidate_isPhoton");
            //     listOfVecVars.push_back("AllFrame_PF_candidate_isNeutralHadron");
            //     listOfVecVars.push_back("AllFrame_PF_candidate_isChargedHadron");

            //     // PUPPI weights
            //     listOfVecVars.push_back("AllFrame_PF_candidate_PUPPIweights");
            // }
        // }
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

    //------------------------------------------------------------------------------
    // Define input tags -----------------------------------------------------------
    //------------------------------------------------------------------------------

    // AK8 Jets
    edm::InputTag ak8JetsTag_;
    //ak8JetsTag_ = edm::InputTag(inputJetColl_, "", "PAT");
    ak8JetsTag_ = edm::InputTag(inputJetColl_, "", "run");
    ak8JetsToken_ = consumes<std::vector<pat::Jet> >(ak8JetsTag_);

    // Sub Jets
    edm::InputTag subJetsTag_;
    subJetsTag_ = edm::InputTag(inputSubJetColl_, "", "run");
    //subJetsTag_ = edm::InputTag("updatedPatJetsTransientCorrectedSoftDropSubjetsPFAK8DF", "", "run");
    //subJetsTag_ = edm::InputTag("selectedUpdatedPatJetsSoftDropSubjetsPFAK8DF", "SubJets", "run");
    //subJetsTag_ = edm::InputTag("selectedUpdatedPatJetsSoftDropSubjetsPFAK8DF", "", "PAT");
    subJetsToken_ = consumes<std::vector<pat::Jet> >(subJetsTag_);

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

    // Find objects corresponding to the token and link to the handle

    /*
    Handle<pat::JetCollection> ak8JetsCollection;
    iEvent.getByToken(ak8JetsToken_, ak8JetsCollection);
    pat::JetCollection ak8Jets = *ak8JetsCollection.product();
    */

    Handle< std::vector<pat::Jet> > ak8JetsCollection;
    iEvent.getByToken(ak8JetsToken_, ak8JetsCollection);
    vector<pat::Jet> ak8Jets = *ak8JetsCollection.product();

    Handle< std::vector<pat::Jet> > subJetsCollection;
    iEvent.getByToken(subJetsToken_, subJetsCollection);
    vector<pat::Jet> subJets = *subJetsCollection.product();

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
            pdgID = 25; break;
        case 2:
            pdgID = 6; break;
        case 3:
            pdgID = 24; break;
        case 4:
            pdgID = 23; break;
        case 5:
            pdgID = 5; break;
        default:
            pdgID = -99; break;
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

    // Match Subjets
    std::map<int, std::vector<int>> subjetMatch = matchSubjets(ak8Jets, subJets);
    // match using dictionary: dict[ak8jetindex] = vectorofupdatedIndices [0=lead, 1=sublead]

    int thisAK8JetIndex = -1;
    for (vector<pat::Jet>::const_iterator jetBegin = ak8Jets.begin(), jetEnd = ak8Jets.end(), ijet = jetBegin; ijet != jetEnd; ++ijet){
        ++thisAK8JetIndex;
        bool GenMatching = false;
        daughtersOfJet.clear();
        boostedDaughters.clear();
        restJets.clear();
        TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );

        // if(ijet->subjets("SoftDropPuppi").size() >=2 && ijet->numberOfDaughters() > 2 && ijet->pt() >= 500 && fabs(ijet->eta()) < 2.4 &&ijet->userFloat("ak8PFJetsPuppiSoftDropMass") > 10) {
        // if(ijet->subjets("SoftDropPuppi").size() >=2 && ijet->numberOfDaughters() > 2 && ijet->pt() >= 500 && fabs(ijet->eta()) < 2.4) {
        if(ijet->subjets("SoftDropPuppi").size() >=2 && ijet->numberOfDaughters() > 2 
            && ijet->pt() >= 500 && ijet->pt() <= 3500 && fabs(ijet->eta()) < 2.4 
            && ijet->userFloat("ak8PFJetsPuppiSoftDropMass") > 0.25) {

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
                // treeVars["nJets"] = ak8Jets.size();
                if (storeJetVariables(treeVars, ijet, subJets, subjetMatch[thisAK8JetIndex]) == false) goto endjetloop;
                
                // Secondary Vertex Variables
                storeSecVertexVariables(treeVars, jetVecVars, jet, secVertices);

                // Get all of the Jet's daughters
                getJetDaughters(daughtersOfJet, ijet);
                if (daughtersOfJet.size() < 3) continue;

                for (unsigned int imass=0; imass < restMasses.size(); imass++) {
                    // Calculate all rest frame variables, skip both loops if frame doesn't work
                    if (calcBESvariables(treeVars, daughtersOfJet, boostedDaughters, ijet, restJets, restMasses[imass]) == false) goto endjetloop;
                }

                // store daughters, rest frame daughters, and rest frame jets
                if(storeDaughters == true){
                    storeJetDaughters(daughtersOfJet, ijet, boostedDaughters, restJets, restMasses, jetVecVars );
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
    }
    // Delete vector
    daughtersOfJet.clear();
    boostedDaughters.clear();
    restJets.clear();
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
