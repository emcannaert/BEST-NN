//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// BESTtoolbox.cpp -------------------------------------------------------------------
//========================================================================================
// C++ file containing functions for use with CMS EDAnalyzer and EDProducer --------------
//////////////////////////////////////////////////////////////////////////////////////////

#include "BESTtoolbox.h"

//========================================================================================
// Calculate Legendre Polynomials --------------------------------------------------------
//----------------------------------------------------------------------------------------
// Simple Legendre polynomial function that can calculate up to order 4 ------------------
// Inputs: argument of the polynomial and order desired ----------------------------------
//----------------------------------------------------------------------------------------

float LegendreP(float x, int order){
    if (order == 0) return 1;
    else if (order == 1) return x;
    else if (order == 2) return 0.5*(3*x*x - 1);
    else if (order == 3) return 0.5*(5*x*x*x - 3*x);
    else if (order == 4) return 0.125*(35*x*x*x*x - 30*x*x + 3);
    else return 0;
}

//========================================================================================
// Calculate Fox Wolfram Moments ---------------------------------------------------------
//----------------------------------------------------------------------------------------
// This function calculates the Fox Wolfram moments for jet constituents -----------------
// in various rest frames. ---------------------------------------------------------------
// Inputs: particles (jet constiuents boosted to rest frame) and empty array that --------
//         that will store the FW moments ------------------------------------------------
//----------------------------------------------------------------------------------------

int FWMoments(std::vector<TLorentzVector> particles, double (&outputs)[5] ){

    // get number of particles to loop over
    int numParticles = particles.size();

    // get energy normalization for the FW moments
    float s = 0.0;
    for(int i = 0; i < numParticles; i++){
        s += particles[i].E();
    }

    float H0 = 0.0;
    float H4 = 0.0;
    float H3 = 0.0;
    float H2 = 0.0;
    float H1 = 0.0;

    for (int i = 0; i < numParticles; i++){

        for (int j = i; j < numParticles; j++){

            // calculate cos of jet constituent angles
            float costh = ( particles[i].Px() * particles[j].Px() + particles[i].Py() * particles[j].Py()
                                        + particles[i].Pz() * particles[j].Pz() ) / ( particles[i].P() * particles[j].P() );
            float w1 = particles[i].P();
            float w2 = particles[j].P();

            // calculate legendre polynomials of jet constiteuent angles
            float fw0 = LegendreP(costh, 0);
            float fw1 = LegendreP(costh, 1);
            float fw2 = LegendreP(costh, 2);
            float fw3 = LegendreP(costh, 3);
            float fw4 = LegendreP(costh, 4);

            // calculate the Fox Wolfram moments
            H0 += w1 * w2 * fw0;
            H1 += w1 * w2 * fw1;
            H2 += w1 * w2 * fw2;
            H3 += w1 * w2 * fw3;
            H4 += w1 * w2 * fw4;
        }
    }

    // Normalize the Fox Wolfram moments
    if (H0 == 0) H0 += 0.001;      // to prevent dividing by zero
    outputs[0] = (H0);
    outputs[1] = (H1 / H0);
    outputs[2] = (H2 / H0);
    outputs[3] = (H3 / H0);
    outputs[4] = (H4 / H0);

    return 0;
}

//========================================================================================
// Get All Jet Constituents --------------------------------------------------------------
//----------------------------------------------------------------------------------------
// This gets all the jet constituents (daughters) and stores them as a standard ----------
// vector --------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
void getJetDaughters(std::vector<reco::Candidate * > &daughtersOfJet, std::vector<pat::Jet>::const_iterator jet){
    // First get all daughters for the first Soft Drop Subjet
    for (unsigned int i = 0; i < jet->daughter(0)->numberOfDaughters(); i++){
        if (jet->daughter(0)->daughter(i)->pt() < 0.5) continue;
        daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(0)->daughter(i) );
    }
    // Get all daughters for the second Soft Drop Subjet
    for (unsigned int i = 0; i < jet->daughter(1)->numberOfDaughters(); i++){
        if (jet->daughter(1)->daughter(i)->pt() < 0.5) continue;
        daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(1)->daughter(i));
    }
    // Get all daughters not included in Soft Drop
    for (unsigned int i = 2; i< jet->numberOfDaughters(); i++){
        if (jet->daughter(i)->pt() < 0.5) continue;
        daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(i) );
    }
}

//========================================================================================
// Store Jet Variables -------------------------------------------------------------------
//----------------------------------------------------------------------------------------
// This takes various jet quantaties and stores them on the map used to fill -------------
// the jet tree --------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
void storeJetVariables(std::map<std::string, float> &besVars, std::vector<pat::Jet>::const_iterator jet,
                       int jetColl){
    // pasing a variable with & is pass-by-reference which keeps changes in this func

    // Jet four vector and Soft Drop info
    besVars["jetAK8_phi"]   = jet->phi();
    besVars["jetAK8_eta"]   = jet->eta();
    besVars["jetAK8_pt"]    = jet->pt();
    besVars["jetAK8_mass"]  = jet->mass();
    besVars["bDisc"]        = jet->bDiscriminator("pfDeepCSVJetTags:probb") + jet->bDiscriminator("pfDeepCSVJetTags:probbb");
    besVars["bDisc_probb"]  = jet->bDiscriminator("pfDeepCSVJetTags:probb");
    besVars["bDisc_probbb"] = jet->bDiscriminator("pfDeepCSVJetTags:probbb");

    // Deep AK8
    // std::cout<<"This Jet Scores: "<<jet->bDiscriminator("pfDeepCSVJetTags:probb")<<", "<<jet->bDiscriminator("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvsLight")<<", "<<jet->bDiscriminator("pfBoostedDoubleSecondaryVertexAK8BJetTags")<<", "<<jet->bDiscriminator("pfMassIndependentDeepDoubleBvLJetTags:probQCD")<<", "<<jet->bDiscriminator("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:WvsQCD")<<std::endl;
    // std::cout<<"This Jet Scores, Part 2: "<<jet->bDiscriminator("pfBoostedDoubleSecondaryVertexAK8BJetTags")<<", "<<jet->bDiscriminator("pfMassIndependentDeepDoubleBvLJetTags:probQCD")<<", "<<jet->bDiscriminator("pfMassIndependentDeepDoubleBvLJetTags:probHbb")<<", "<<jet->bDiscriminator("pfMassIndependentDeepDoubleCvLJetTags:probQCD")<<", "<<jet->bDiscriminator("pfMassIndependentDeepDoubleCvLJetTags:probHcc")<<", "<<jet->bDiscriminator("pfMassIndependentDeepDoubleCvBJetTags:probHbb")<<", "<<jet->bDiscriminator("pfMassIndependentDeepDoubleCvBJetTags:probHcc")<<std::endl;
    // std::cout<<"This Jet Scores, Part 3: "<<jet->bDiscriminator("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvsLight")<<", "<<jet->bDiscriminator("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ccvsLight")<<", "<<jet->bDiscriminator("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:TvsQCD")<<", "<<jet->bDiscriminator("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHccvsQCD")<<", "<<jet->bDiscriminator("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:WvsQCD")<<", "<<jet->bDiscriminator("pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHbbvsQCD")<<std::endl;
    besVars["jetAK8_deepAK8_rawL"] = jet->bDiscriminator("pfDeepBoostedJetTags:probQCDothers");
    besVars["jetAK8_deepAK8_rawC"] = jet->bDiscriminator("pfDeepBoostedJetTags:probQCDcc") + jet->bDiscriminator("pfDeepBoostedJetTags:probQCDc");
    besVars["jetAK8_deepAK8_rawB"] = jet->bDiscriminator("pfDeepBoostedJetTags:probQCDbb") + jet->bDiscriminator("pfDeepBoostedJetTags:probQCDb");
    besVars["jetAK8_deepAK8_rawW"] = jet->bDiscriminator("pfDeepBoostedJetTags:probWcq") + jet->bDiscriminator("pfDeepBoostedJetTags:probWqq");
    besVars["jetAK8_deepAK8_rawZ"] = jet->bDiscriminator("pfDeepBoostedJetTags:probZbb") + jet->bDiscriminator("pfDeepBoostedJetTags:probZcc") + jet->bDiscriminator("pfDeepBoostedJetTags:probZqq");
    besVars["jetAK8_deepAK8_rawH"] = jet->bDiscriminator("pfDeepBoostedJetTags:probHbb") + jet->bDiscriminator("pfDeepBoostedJetTags:probHcc") + jet->bDiscriminator("pfDeepBoostedJetTags:probHqqqq");
    besVars["jetAK8_deepAK8_rawT"] = jet->bDiscriminator("pfDeepBoostedJetTags:probTbcq") + jet->bDiscriminator("pfDeepBoostedJetTags:probTbqq");
    besVars["jetAK8_deepAK8_rawmax"] = std::max({besVars["jetAK8_deepAK8_rawL"],besVars["jetAK8_deepAK8_rawC"],besVars["jetAK8_deepAK8_rawB"],besVars["jetAK8_deepAK8_rawW"],besVars["jetAK8_deepAK8_rawZ"],besVars["jetAK8_deepAK8_rawH"],besVars["jetAK8_deepAK8_rawT"]});        
    besVars["jetAK8_deepAK8_dnn_Largest"] = 10;

    float epsilon = 1e-4;
    // J, T, H, Z, W, B, C = 0, 1, 2, 3, 4, 5, 6
    if (besVars["jetAK8_deepAK8_rawmax"] - besVars["jetAK8_deepAK8_rawL"] < epsilon) besVars["jetAK8_deepAK8_dnn_Largest"] = 0;
    else if (besVars["jetAK8_deepAK8_rawmax"] - besVars["jetAK8_deepAK8_rawT"] < epsilon) besVars["jetAK8_deepAK8_dnn_Largest"] = 1;
    else if (besVars["jetAK8_deepAK8_rawmax"] - besVars["jetAK8_deepAK8_rawH"] < epsilon) besVars["jetAK8_deepAK8_dnn_Largest"] = 2;
    else if (besVars["jetAK8_deepAK8_rawmax"] - besVars["jetAK8_deepAK8_rawZ"] < epsilon) besVars["jetAK8_deepAK8_dnn_Largest"] = 3;
    else if (besVars["jetAK8_deepAK8_rawmax"] - besVars["jetAK8_deepAK8_rawW"] < epsilon) besVars["jetAK8_deepAK8_dnn_Largest"] = 4;
    else if (besVars["jetAK8_deepAK8_rawmax"] - besVars["jetAK8_deepAK8_rawB"] < epsilon) besVars["jetAK8_deepAK8_dnn_Largest"] = 5;
    else if (besVars["jetAK8_deepAK8_rawmax"] - besVars["jetAK8_deepAK8_rawC"] < epsilon) besVars["jetAK8_deepAK8_dnn_Largest"] = 6;
    else besVars["jetAK8_deepAK8_dnn_Largest"] = 10;
    // MD Deep AK8
    besVars["jetAK8_deepAK8MD_rawL"] = jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probQCDothers");
    besVars["jetAK8_deepAK8MD_rawC"] = jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probQCDcc") + jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probQCDc");
    besVars["jetAK8_deepAK8MD_rawB"] = jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probQCDbb") + jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probQCDb");
    besVars["jetAK8_deepAK8MD_rawW"] = jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probWcq") + jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probWqq");
    besVars["jetAK8_deepAK8MD_rawZ"] = jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probZbb") + jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probZcc") + jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probZqq");
    besVars["jetAK8_deepAK8MD_rawH"] = jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probHbb") + jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probHcc") + jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probHqqqq");
    besVars["jetAK8_deepAK8MD_rawT"] = jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probTbcq") + jet->bDiscriminator("pfMassDecorrelatedDeepBoostedJetTags:probTbqq");
    besVars["jetAK8_deepAK8MD_rawmax"] = std::max({besVars["jetAK8_deepAK8MD_rawL"],besVars["jetAK8_deepAK8MD_rawC"],besVars["jetAK8_deepAK8MD_rawB"],besVars["jetAK8_deepAK8MD_rawW"],besVars["jetAK8_deepAK8MD_rawZ"],besVars["jetAK8_deepAK8MD_rawH"],besVars["jetAK8_deepAK8MD_rawT"]});        
    besVars["jetAK8_deepAK8MD_dnn_Largest"] = 10;
    // J, T, H, Z, W, B, C = 0, 1, 2, 3, 4, 5, 6
    if (besVars["jetAK8_deepAK8MD_rawmax"] - besVars["jetAK8_deepAK8MD_rawL"] < epsilon) besVars["jetAK8_deepAK8MD_dnn_Largest"] = 0;
    else if (besVars["jetAK8_deepAK8MD_rawmax"] - besVars["jetAK8_deepAK8MD_rawT"] < epsilon) besVars["jetAK8_deepAK8MD_dnn_Largest"] = 1;
    else if (besVars["jetAK8_deepAK8MD_rawmax"] - besVars["jetAK8_deepAK8MD_rawH"] < epsilon) besVars["jetAK8_deepAK8MD_dnn_Largest"] = 2;
    else if (besVars["jetAK8_deepAK8MD_rawmax"] - besVars["jetAK8_deepAK8MD_rawZ"] < epsilon) besVars["jetAK8_deepAK8MD_dnn_Largest"] = 3;
    else if (besVars["jetAK8_deepAK8MD_rawmax"] - besVars["jetAK8_deepAK8MD_rawW"] < epsilon) besVars["jetAK8_deepAK8MD_dnn_Largest"] = 4;
    else if (besVars["jetAK8_deepAK8MD_rawmax"] - besVars["jetAK8_deepAK8MD_rawB"] < epsilon) besVars["jetAK8_deepAK8MD_dnn_Largest"] = 5;
    else if (besVars["jetAK8_deepAK8MD_rawmax"] - besVars["jetAK8_deepAK8MD_rawC"] < epsilon) besVars["jetAK8_deepAK8MD_dnn_Largest"] = 6;
    else besVars["jetAK8_deepAK8MD_dnn_Largest"] = 10;

    // Store Jet Charge = (1/sum{pT^0.6}) * sum{q * pT^0.6}
    float jetCharge = 0;
    float jetChargeNorm = 0;
    for(int subJetIndex = 0; subJetIndex<(int)jet->numberOfDaughters(); subJetIndex++) {
        float subJetPt = jet->daughter(subJetIndex)->pt();
        jetCharge += pow(subJetPt,0.6)*jet->daughter(subJetIndex)->charge();
        jetChargeNorm += pow(subJetPt,0.6);
    }
    if (jetChargeNorm > 0) {
        jetCharge = jetCharge/jetChargeNorm;
    } else {
        jetCharge = -99;
    }
    besVars["jetAK8_charge"] = jetCharge;


    // Store Subjettiness info
    if(jetColl == 0){ // CHS jets
        besVars["jetAK8_Tau4"]  = jet->userFloat("NjettinessAK8CHS:tau4");  //important for H->WW jets
        besVars["jetAK8_Tau3"]  = jet->userFloat("NjettinessAK8CHS:tau3");
        besVars["jetAK8_Tau2"]  = jet->userFloat("NjettinessAK8CHS:tau2");
        besVars["jetAK8_Tau1"]  = jet->userFloat("NjettinessAK8CHS:tau1");
        besVars["jetAK8_Tau21"] = jet->userFloat("NjettinessAK8CHS:tau2") / jet->userFloat("NjettinessAK8CHS:tau1");
        besVars["jetAK8_Tau32"] = jet->userFloat("NjettinessAK8CHS:tau3") / jet->userFloat("NjettinessAK8CHS:tau2");
        besVars["jetAK8_SoftDropMass"] = jet->userFloat("ak8PFJetsCHSValueMap:ak8PFJetsCHSSoftDropMass");
    }
    if(jetColl == 1){ // PUPPI jets
        besVars["jetAK8_Tau4"]  = jet->userFloat("NjettinessAK8Puppi:tau4");  //important for H->WW jets
        besVars["jetAK8_Tau3"]  = jet->userFloat("NjettinessAK8Puppi:tau3");
        besVars["jetAK8_Tau2"]  = jet->userFloat("NjettinessAK8Puppi:tau2");
        besVars["jetAK8_Tau1"]  = jet->userFloat("NjettinessAK8Puppi:tau1");
        besVars["jetAK8_Tau21"] = jet->userFloat("NjettinessAK8Puppi:tau2") / jet->userFloat("NjettinessAK8Puppi:tau1");
        besVars["jetAK8_Tau32"] = jet->userFloat("NjettinessAK8Puppi:tau3") / jet->userFloat("NjettinessAK8Puppi:tau2");
        besVars["jetAK8_SoftDropMass"] = jet->userFloat("ak8PFJetsPuppiSoftDropMass");
        auto subjets = jet->subjets("SoftDropPuppi");
        if (subjets.size() < 2){
            std::cout << "This will exit, not enough subjets" << std::endl;
            exit(1);
        }
        if (!subjets[0]){
            std::cout << "This will exit, invalid subjet 0" << std::endl;
            exit(1);
        }
        if (!subjets[1]){
            std::cout << "This will exit, invalid subjet 1" << std::endl;
                exit(1);
        }
        // Fill leading subjet bDisc variables, and get maximum bDisc values
        double maxbDisc = 0;
        double imaxbDisc;
        for (unsigned int isubjet=0; isubjet < subjets.size(); isubjet++) {
            double bDiscVal = subjets[isubjet]->bDiscriminator("pfDeepCSVJetTags:probb") + subjets[isubjet]->bDiscriminator("pfDeepCSVJetTags:probbb");
            
            // Find max bDisc value and index
            if (bDiscVal > maxbDisc) { 
                maxbDisc  = bDiscVal;
                imaxbDisc = isubjet;
            }

            // Fill BES var for leading two subjets
            if (isubjet <= 1) { // Only triggers for first 2 iterations of loop, the leading subjets
                std::string leadingSubJet = std::to_string(isubjet + 1); // this is either 1 or 2, the leading subjets, which correspond to index 0 and 1 for isubjet
                besVars["bDisc"+leadingSubJet]           = bDiscVal;
                besVars["bDisc"+leadingSubJet+"_probb"]  = subjets[isubjet]->bDiscriminator("pfDeepCSVJetTags:probb");
                besVars["bDisc"+leadingSubJet+"_probbb"] = subjets[isubjet]->bDiscriminator("pfDeepCSVJetTags:probbb");
            }       
        }
        besVars["bDiscSubJet_Max"] = maxbDisc;
        besVars["bDiscSubJet_Max_index"] = imaxbDisc; // indexes from 0
        // separate these like the above abbott
        // maximum subjet CSV value -> loop through all subjets, always store 0 and 1, but do them all and store the index + value of largest probb + probbb
        // sep branch bDisc_Max and bDisc_Max_i
        // besVars["bDisc1"] = subjets[0]->bDiscriminator("pfDeepCSVJetTags:probb") + subjets[0]->bDiscriminator("pfDeepCSVJetTags:probbb");
        // besVars["bDisc2"] = subjets[1]->bDiscriminator("pfDeepCSVJetTags:probb") + subjets[1]->bDiscriminator("pfDeepCSVJetTags:probbb");
    }
}

//========================================================================================
// Store Secondary Vertex Information ----------------------------------------------------
//----------------------------------------------------------------------------------------
// This takes various secondary vertex quantities and stores them on the map -------------
// used to fill the tree -----------------------------------------------------------------
//----------------------------------------------------------------------------------------
void storeSecVertexVariables(std::map<std::string, float> &besVars,
                             std::map<std::string, std::vector<float> > &jetVecVars, TLorentzVector jet,
                             std::vector<reco::VertexCompositePtrCandidate> secVertices){
    int numMatched = 0; // counts number of secondary vertices
    for(std::vector<reco::VertexCompositePtrCandidate>::const_iterator vertBegin = secVertices.begin(),
                            vertEnd = secVertices.end(), ivert = vertBegin; ivert != vertEnd; ivert++){
        TLorentzVector vert(ivert->px(), ivert->py(), ivert->pz(), ivert->energy() );
        // match vertices to jet
        if(jet.DeltaR(vert) < 0.8 ){
            numMatched++;
            // save secondary vertex info for the first three sec vertices
            jetVecVars["SV_pt"].push_back(ivert->pt() );
            jetVecVars["SV_eta"].push_back(ivert->eta() );
            jetVecVars["SV_phi"].push_back(ivert->phi() );
            jetVecVars["SV_mass"].push_back(ivert->mass() );
            jetVecVars["SV_nTracks"].push_back(ivert->numberOfDaughters() );
            jetVecVars["SV_chi2"].push_back(ivert->vertexChi2() );
            jetVecVars["SV_Ndof"].push_back(ivert->vertexNdof() );
        }
    }
    // abbott start to process samples to prepare for BEST training
    // big task: format converter->SV stuff disabled there
    // SV number is variable, NN needs a fixed amount of inputs...what do?
    // plot nSecondaryVertices, if managable choose 10 or 5 and pad out zeroes
    // doo root tree scans, root -b file.root, get tree, tree->Scan("nSecVert:SV_pt","cut") <- check that we are seeing all the SV that are there
    besVars["nSecondaryVertices"] = numMatched;
}

//========================================================================================
// Calculate BEST Rest Frame Variables ---------------------------------------------------
//----------------------------------------------------------------------------------------
// This boosts an ak8 jet (and all of its constituents) into heavy object rest frame -----
// and then uses it to calculate FoxWolfram moments, Event Shape Variables, --------------
// and asymmetry variables ---------------------------------------------------------------
// If requirements not satisfied to proceed return false. Else finish by returning true --
//----------------------------------------------------------------------------------------

bool calcBESvariables(std::map<std::string, float> &besVars, std::vector<reco::Candidate *> &daughtersOfJet,
                      std::map<std::string, std::vector<TLorentzVector> > &boostedDaughters,
                      std::vector<pat::Jet>::const_iterator jet, std::map<std::string, std::vector<fastjet::PseudoJet> > &restJets,
                      std::string mass){
                    //   int mass){

    // get 4 vector for heavy object rest frame
    typedef reco::Candidate::PolarLorentzVector fourv;
    fourv thisJet = jet->polarP4();
    TLorentzVector thisJetLV(0.,0.,0.,0.);
    // Turn this into case statement? Update after BEST testing
    if      ( mass == "ak8" )           thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), (float)jet->mass() );
    else if ( mass == "ak8SoftDrop" )   thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), jet->userFloat("ak8PFJetsPuppiSoftDropMass") );
    else if ( mass == "Bottom" )        thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), 6.0 );
    else if ( mass == "W" )             thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), 80.4 );
    else if ( mass == "Z" )             thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), 91.2 );
    else if ( mass == "Higgs" )         thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), 125.0 );
    else if ( mass == "Top" )           thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), 172.5 );
    else if ( mass == "Lab")            thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), thisJet.M() );
    else                                thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), std::stof(mass) ); // The "GeV" automatically gets trimmed when converting to float

    
    
    // thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), (float)mass );
    // Possibly adjust pT as well? think about physics (and exp) like assume measuring higgs, see this mass, what correction to pT to get best fit? dif for dif signals 

    std::vector<TLorentzVector> particles;
    std::vector<math::XYZVector> particles2;
    std::vector<reco::LeafCandidate> particles3;
    std::vector<fastjet::PseudoJet> FJparticles;

    // 4 vectors to be filled with reclustered jet additions
    TLorentzVector jet12LV(0.,0.,0.,0.);
    TLorentzVector jet13LV(0.,0.,0.,0.);
    TLorentzVector jet23LV(0.,0.,0.,0.);
    TLorentzVector jet1234LV(0.,0.,0.,0.);

    double sumPz = 0;
    double sumP = 0;
    // Boost to object rest frame
    for(unsigned int i = 0; i < daughtersOfJet.size(); i++){
        // Do not include low pT particles
        if (daughtersOfJet[i]->pt() < 0.5) continue;

        // Create 4 vector to boost to Higgs frame
        TLorentzVector thisParticleLV( daughtersOfJet[i]->px(), daughtersOfJet[i]->py(), daughtersOfJet[i]->pz(), daughtersOfJet[i]->energy() );
        // Boost to heavy object rest frame
        thisParticleLV.Boost( -thisJetLV.BoostVector() );

        // Now that PF candidates are stored, make the boost axis the Z-axis
        // Important for BES variables
        pboost( thisJetLV.Vect(), thisParticleLV.Vect(), thisParticleLV);

        particles.push_back( thisParticleLV );
        particles2.push_back( math::XYZVector( thisParticleLV.X(), thisParticleLV.Y(), thisParticleLV.Z() ));
        particles3.push_back( reco::LeafCandidate(+1, reco::Candidate::LorentzVector( thisParticleLV.X(), thisParticleLV.Y(),
                                                                                        thisParticleLV.Z(), thisParticleLV.T() ) ));
        FJparticles.push_back( fastjet::PseudoJet( thisParticleLV.X(), thisParticleLV.Y(), thisParticleLV.Z(), thisParticleLV.T() ) );

        // Sum rest frame momenta for asymmetry calculation
        sumPz += thisParticleLV.Pz();
        sumP  += abs( thisParticleLV.P() );
    }

    // std::string frame = std::to_string(mass)+"GeV";
    // std::string frame = mass+"GeV";
    std::string frame = mass;
    
    // Jet Asymmetry
    besVars["asymmetry_"+frame] = sumPz/sumP;

    // Fox Wolfram Moments
    double fwm[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
    FWMoments( particles, fwm);
    besVars["FoxWolfH1_"+frame] = fwm[1];
    besVars["FoxWolfH2_"+frame] = fwm[2];
    besVars["FoxWolfH3_"+frame] = fwm[3];
    besVars["FoxWolfH4_"+frame] = fwm[4];

    // Event Shape Variables
    EventShapeVariables eventShapes( particles2 );
    Thrust thrustCalculator( particles3.begin(), particles3.end() );
    besVars["isotropy_"+frame]   = eventShapes.isotropy();
    besVars["sphericity_"+frame] = eventShapes.sphericity();
    besVars["aplanarity_"+frame] = eventShapes.aplanarity();
    besVars["thrust_"+frame]     = thrustCalculator.thrust();

    // Recluster the jets in the heavy object rest frame
    fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, 0.4);
    fastjet::ClusterSequence cs(FJparticles, jet_def);
    // std::vector<fastjet::PseudoJet> jetsFJ = sorted_by_pt(cs.inclusive_jets(20.0));
    // Changed to 0.0 here, the 20.0 cuts on pT relative to a meaningless axis
    // std::vector<fastjet::PseudoJet> jetsFJ = sorted_by_pt(cs.inclusive_jets(0.0));
    std::vector<fastjet::PseudoJet> jetsFJ = sorted_by_E(cs.inclusive_jets(0.0));
    restJets[frame+"Frame"] = jetsFJ;

    // Store reclustered jet info
    for(unsigned int i = 0; i < jetsFJ.size(); i++){
        // make a TLorentzVector for the current clustered rest frame jet
        TLorentzVector iJetLV(jetsFJ[i].px(), jetsFJ[i].py(), jetsFJ[i].pz(), jetsFJ[i].e() );

        // get rest frame jet four vector combinations
        switch(i){
            case 0:
                // jet12LV   = jet12LV   + iJetLV;
                // jet13LV   = jet13LV   + iJetLV;
                // jet1234LV = jet1234LV + iJetLV;
                jet12LV   = iJetLV;
                jet13LV   = iJetLV;
                jet1234LV = iJetLV;
                break;
            case 1:
                besVars["jet12_DeltaCosTheta_"+frame]   = (jet12LV.Vect()).Dot(iJetLV.Vect()) / (jet12LV.Vect().Mag() * iJetLV.Vect().Mag());
                jet12LV   = jet12LV   + iJetLV;
                // jet23LV   = jet23LV   + iJetLV;
                jet23LV   = iJetLV;
                jet1234LV = jet1234LV + iJetLV;
                break;
            case 2:
                besVars["jet13_DeltaCosTheta_"+frame]   = (jet13LV.Vect()).Dot(iJetLV.Vect()) / (jet13LV.Vect().Mag() * iJetLV.Vect().Mag());
                jet13LV   = jet13LV   + iJetLV;
                besVars["jet23_DeltaCosTheta_"+frame]   = (jet23LV.Vect()).Dot(iJetLV.Vect()) / (jet23LV.Vect().Mag() * iJetLV.Vect().Mag());
                jet23LV   = jet23LV   + iJetLV;
                jet1234LV = jet1234LV + iJetLV;
                break;
            case 3:
                jet1234LV = jet1234LV + iJetLV;
                break;
        }
    }
    
    // Store boosted candidates for rest frames
    boostedDaughters[frame+"Frame"] = particles;

    // Store reclustered jet mass/angle combinations
    besVars["jet12_mass_"+frame]        = jet12LV.M();
    besVars["jet13_mass_"+frame]        = jet13LV.M();
    besVars["jet23_mass_"+frame]        = jet23LV.M();
    besVars["jet1234_mass_"+frame]      = jet1234LV.M();
    besVars["jet12_CosTheta_"+frame]    = jet12LV.CosTheta();
    besVars["jet13_CosTheta_"+frame]    = jet13LV.CosTheta();
    besVars["jet23_CosTheta_"+frame]    = jet23LV.CosTheta();
    besVars["jet1234_CosTheta_"+frame]  = jet1234LV.CosTheta();
    besVars["nJets_"+frame]             = jetsFJ.size();

    return true;
}

//========================================================================================
// Store Jet Constituents ----------------------------------------------------------------
//----------------------------------------------------------------------------------------
// This stores all the jet constituents in a vector corresponding to a jetTree so these --
//  variables can be used in training BEST -----------------------------------------------
//----------------------------------------------------------------------------------------

void storeJetDaughters(std::vector<reco::Candidate * > &daughtersOfJet, std::vector<pat::Jet>::const_iterator jet,
                       std::map<std::string, std::vector<TLorentzVector> > &boostedDaughters,
                    //    std::map<std::string, std::vector<fastjet::PseudoJet> > &restJets, std::vector<int> restMasses,
                       std::map<std::string, std::vector<fastjet::PseudoJet> > &restJets, std::vector<std::string> restMasses,
                       std::map<std::string, std::vector<float> > &jetVecVars, int jetColl ){
    // loop over lab frame candidates
    for(unsigned int i = 0; i < daughtersOfJet.size(); i++){

        // Do not include low pT particles
        if (daughtersOfJet[i]->pt() < 0.5) continue;

        // Calculate delta eta and phi for the candidates to the center of the jet
        float deltaEta = daughtersOfJet[i]->eta() - jet->eta();
        float deltaPhi = daughtersOfJet[i]->phi() - jet->phi();

        // Calculate pT and Energy logarithms for candidates
        float logpT     = TMath::Log(daughtersOfJet[i]->pt());
        float logEnergy = TMath::Log(daughtersOfJet[i]->energy());

        // Determine particle type, set boolean flags
        int  absPDGID = abs( daughtersOfJet[i]->pdgId() );
        bool isElectron      = false;
        bool isMuon          = false;
        bool isPhoton        = false;
        bool isNeutralHadron = false; // In our analysis, the only neutral hadrons we see are K longs.
        bool isChargedHadron = false; // In our analysis, the only charged hadrons we see are pions.

        // Set flags:
        switch(absPDGID){
        case 11: // Check is candidate is an electron or positron
            isElectron      = true; break;
        case 13: // Check if candidate is a muon or antimuon
            isMuon          = true; break;
        case 22: // Check if candidate is a photon
            isPhoton        = true; break;
        case 130: // Check if candidate is K long
            isNeutralHadron = true; break;
        case 211: // Check if candidate is pion or antipion
            isChargedHadron = true; break; 
        }

        // Store the candidate
        jetVecVars["LabFrame_PF_candidate_px"].push_back(daughtersOfJet[i]->px() );
        jetVecVars["LabFrame_PF_candidate_py"].push_back(daughtersOfJet[i]->py() );
        jetVecVars["LabFrame_PF_candidate_pz"].push_back(daughtersOfJet[i]->pz() );
        jetVecVars["LabFrame_PF_candidate_energy"].push_back(daughtersOfJet[i]->energy() );

        jetVecVars["AllFrame_PF_candidate_charge"].push_back(daughtersOfJet[i]->charge() );
        jetVecVars["AllFrame_PF_candidate_pdgId"].push_back(daughtersOfJet[i]->pdgId() );
        jetVecVars["AllFrame_PF_candidate_abspdgId"].push_back(absPDGID );
        jetVecVars["AllFrame_PF_candidate_isElectron"].push_back(isElectron );
        jetVecVars["AllFrame_PF_candidate_isMuon"].push_back(isMuon );
        jetVecVars["AllFrame_PF_candidate_isPhoton"].push_back(isPhoton );
        jetVecVars["AllFrame_PF_candidate_isNeutralHadron"].push_back(isNeutralHadron );
        jetVecVars["AllFrame_PF_candidate_isChargedHadron"].push_back(isChargedHadron );

        jetVecVars["LabFrame_PF_candidate_deltaEta"].push_back(deltaEta );
        jetVecVars["LabFrame_PF_candidate_deltaPhi"].push_back(deltaPhi );
        jetVecVars["LabFrame_PF_candidate_deltaR"].push_back( TMath::Sqrt( TMath::Sq(deltaEta) + TMath::Sq(deltaPhi) ) ); // Angular separation between the candidate and the jet axis
        jetVecVars["LabFrame_PF_candidate_logpT"].push_back(logpT );
        jetVecVars["LabFrame_PF_candidate_logEnergy"].push_back(logEnergy );
        jetVecVars["LabFrame_PF_candidate_logpTRatio"].push_back(logpT - TMath::Log(jet->pt()) ); // Logarithm of the candidate's pT relative to the jet pT
        jetVecVars["LabFrame_PF_candidate_logEnergyRatio"].push_back(logEnergy - TMath::Log(jet->energy()) ); // Logarithm of the candidate's energy releative to the jet energy

        // PUPPI weights for puppi jets
        if (jetColl == 1){
            pat::PackedCandidate *iparticle = (pat::PackedCandidate *) daughtersOfJet[i];
            if(!iparticle){
                std::cout<<"ERROR: The PF candidate did not get properly converted to PackedCandidate in lab frame"<<std::endl;
                std::cout<<" 'Transfiguration is some of the most dangerous and complex magic!'"<<std::endl;
                exit(1);
            }
            // jetVecVars["PUPPI_Weights"].push_back( iparticle->puppiWeight() );
            jetVecVars["AllFrame_PF_candidate_PUPPIweights"].push_back( iparticle->puppiWeight() );
        }
    }

    // loop over rest frames
    for(unsigned int iFrame = 0; iFrame < restMasses.size(); iFrame++){

        // std::string frame = std::to_string(restMasses[iFrame]);
        std::string frame = restMasses[iFrame];
        // loop over candidates in the rest frame
        for(auto icand = boostedDaughters[frame+"Frame"].begin(); icand != boostedDaughters[frame+"Frame"].end(); icand++){
            // Calculate delta eta and phi for the candidates to the center of the jet
            float deltaEta = icand->Eta() - jet->eta();
            float deltaPhi = icand->Phi() - jet->phi();

            // Calculate pT and Energy logarithms for candidates
            float logpT     = TMath::Log(icand->Pt());
            float logEnergy = TMath::Log(icand->E());


            // store the rest frame candidate
            jetVecVars[frame+"Frame_PF_candidate_px"].push_back(icand->Px() );
            jetVecVars[frame+"Frame_PF_candidate_py"].push_back(icand->Py() );
            jetVecVars[frame+"Frame_PF_candidate_pz"].push_back(icand->Pz() );
            jetVecVars[frame+"Frame_PF_candidate_energy"].push_back(icand->E() );

            jetVecVars[frame+"Frame_PF_candidate_deltaEta"].push_back(deltaEta );
            jetVecVars[frame+"Frame_PF_candidate_deltaPhi"].push_back(deltaPhi );
            jetVecVars[frame+"Frame_PF_candidate_deltaR"].push_back( TMath::Sqrt( TMath::Sq(deltaEta) + TMath::Sq(deltaPhi) ) ); // Angular separation between the candidate and the jet axis
            jetVecVars[frame+"Frame_PF_candidate_logpT"].push_back(logpT );
            jetVecVars[frame+"Frame_PF_candidate_logEnergy"].push_back(logEnergy );
            jetVecVars[frame+"Frame_PF_candidate_logpTRatio"].push_back(logpT - TMath::Log(jet->pt()) ); // Logarithm of the candidate's pT relative to the jet pT
            jetVecVars[frame+"Frame_PF_candidate_logEnergyRatio"].push_back(logEnergy - TMath::Log(jet->energy()) ); // Logarithm of the candidate's energy releative to the jet energy


            // // PUPPI weights for puppi jets
            // if (jetColl == 1){
            //     pat::PackedCandidate *iparticle = (pat::PackedCandidate *) icand;
            //     if(!iparticle){
            //         std::cout<<"ERROR: The PF candidate did not get properly converted to PackedCandidate in frame: " + frame <<std::endl;
            //         std::cout<<" 'Transfiguration is some of the most dangerous and complex magic!'"<<std::endl;
            //         exit(1);
            //     }
            //     jetVecVars[frame+"Frame_PF_candidate_PUPPI_Weights"].push_back( iparticle->puppiWeight() );
            // }
        }

        // loop over rest frame jets
        for(auto ijet = restJets[frame+"Frame"].begin(); ijet != restJets[frame+"Frame"].end(); ijet++){
            // store the rest frame jet information
            jetVecVars[frame+"Frame_jet_px"].push_back(ijet->px());
            jetVecVars[frame+"Frame_jet_py"].push_back(ijet->py());
            jetVecVars[frame+"Frame_jet_pz"].push_back(ijet->pz());
            jetVecVars[frame+"Frame_jet_energy"].push_back(ijet->e());
        }
    }
}

//========================================================================================
// Make boost axis the rest frame z axis -------------------------------------------------
//----------------------------------------------------------------------------------------
// Given jet constituent lab momentum, find momentum relative to beam direction pbeam ----
// plab = Particle 3-vector in Boost Frame -----------------------------------------------
// pbeam = Lab Jet 3-vector --------------------------------------------------------------
//----------------------------------------------------------------------------------------

void pboost( TVector3 pbeam, TVector3 plab, TLorentzVector &pboo ){

    double pl = plab.Dot(pbeam);
    pl *= double(1. / pbeam.Mag());

    // set x axis direction along pbeam x (0,0,1)
    TVector3 pbx;

    pbx.SetX(pbeam.Y());
    pbx.SetY(-pbeam.X());
    pbx.SetZ(0.0);

    pbx *= double(1. / pbx.Mag());

    // set y axis direction along -pbx x pbeam
    TVector3 pby;

    pby = -pbx.Cross(pbeam);
    pby *= double(1. / pby.Mag());

    pboo.SetX((plab.Dot(pbx)));
    pboo.SetY((plab.Dot(pby)));
    pboo.SetZ(pl);

}
