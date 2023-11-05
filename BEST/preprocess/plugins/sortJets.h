#ifndef __SORTJETS_H__
#define __SORTJETS_H__

#include <cmath>
#include <iostream>
#include "TLorentzVector.h"
#include <string>

using namespace std;
#include <iostream>

class sortJets
{	
	private:
		
		void findBestJetComb(void);

		std::vector<TLorentzVector> superJet1;
		std::vector<TLorentzVector> superJet2;
		std::vector<TLorentzVector> miscJets;
		int nMiscJets;
	public:
		sortJets(std::vector<TLorentzVector>,std::vector<TLorentzVector>,std::vector<TLorentzVector>);
		std::vector<TLorentzVector> finalSuperJet1;
		std::vector<TLorentzVector> finalSuperJet2;

};	
#endif
