#include <cmath>
#include <iostream>
#include "TLorentzVector.h"
#include <string>

using namespace std;
#include <iostream>
#include "/uscms_data/d3/cannaert/analysis/CMSSW_10_6_20/src/sortJets.h"

sortJets::sortJets(std::vector<TLorentzVector> SJ1, std::vector<TLorentzVector> SJ2, std::vector<TLorentzVector> _miscJets )
{
	superJet1 = SJ1;
	superJet2 = SJ2;
	miscJets = _miscJets;
	nMiscJets = miscJets.size();
	findBestJetComb();
}

void sortJets::findBestJetComb()
{


	std::vector<TLorentzVector> bestSuperJet1;
	std::vector<TLorentzVector> bestSuperJet2;


	TLorentzVector totSuperJet1(0,0,0,0);
	TLorentzVector totSuperJet2(0,0,0,0);
	for(unsigned int iii = 0;iii<superJet1.size();iii++)
	{
		totSuperJet1+=superJet1[iii];
	}
	for(unsigned int iii = 0;iii<superJet2.size();iii++)
	{
		totSuperJet2+=superJet2[iii];
	}
	double minMassDiff = 9999999999.;
	int useArray[nMiscJets];

	////////////////////////mmisc == 1 /////////////////////////////
	//std::cout << "New event - number of misc jets: " << nMiscJets << " with masses ";
	for(unsigned int iii = 0;iii<superJet1.size();iii++)
	{
		//std::cout << superJet1[iii].M() << " ";
	}
	for(unsigned int iii = 0;iii<superJet2.size();iii++)
	{
		//std::cout << superJet2[iii].M() << " ";
	}
	for(unsigned int iii = 0;iii<miscJets.size();iii++)
	{
		//std::cout << miscJets[iii].M() << " ";
	}	
	//std::cout << std::endl;	
	if(nMiscJets == 0) return;
	else if(nMiscJets == 1)
	{
		for(int aaa=0;aaa<2;aaa++)
		{
			useArray[0]=aaa;
			if( (superJet1.size()==0)&&(superJet2.size() >0) )
			{
				if( (aaa==1) )continue; 
			}
			else if ( (superJet1.size() > 0)&&(superJet2.size() == 0)  )
			{
				if( (aaa==0) )continue;
			}
			TLorentzVector useSuperJet1 = totSuperJet1;
			TLorentzVector useSuperJet2 = totSuperJet2;
			std::vector<TLorentzVector> _useSuperJet1 = superJet1;;
			std::vector<TLorentzVector> _useSuperJet2 = superJet2;;
			for(int iii = 0;iii<nMiscJets;iii++)
			{	
				if(useArray[iii] == 0)
				{
					useSuperJet1+=miscJets[iii];
					_useSuperJet1.push_back(miscJets[iii]);
				}
				else
				{ 
					useSuperJet2+=miscJets[iii];
					_useSuperJet2.push_back(miscJets[iii]);
				}
			}
			//std::cout << "Clustered jet masses: " << useSuperJet1.M() << " " << useSuperJet2.M() << std::endl;
 			if( (abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M()) ) < minMassDiff )
			{
				minMassDiff = abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M());
				bestSuperJet1 = _useSuperJet1;
				bestSuperJet2 = _useSuperJet2;
			}
		}
	}


	////////////////////////mmisc == 2 /////////////////////////////
	else if(nMiscJets == 2)
	{
		for(int aaa=0;aaa<2;aaa++)
		{
			for(int bbb=0;bbb<2;bbb++)
			{
				if( (superJet1.size()==0)&&(superJet2.size() >0) )
				{
					if( (aaa==1) && (aaa==bbb) )continue; 
				}
				else if ( (superJet1.size() > 0)&&(superJet2.size() == 0)  )
				{
					if( (aaa==0) && (aaa==bbb) )continue;
				}
				else if ( (superJet1.size()==0)&&(superJet2.size() == 0) )
				{
					if((aaa==bbb)) continue;
				} 
				useArray[0]=aaa;useArray[1]=bbb;
				TLorentzVector useSuperJet1 = totSuperJet1;
				TLorentzVector useSuperJet2 = totSuperJet2;
				std::vector<TLorentzVector> _useSuperJet1 = superJet1;
				std::vector<TLorentzVector> _useSuperJet2 = superJet2;
				for(int iii = 0;iii<nMiscJets;iii++)
				{	
					if(useArray[iii] == 0)
					{
						useSuperJet1+=miscJets[iii];
						_useSuperJet1.push_back(miscJets[iii]);
					}
					else
					{ 
						useSuperJet2+=miscJets[iii];
						_useSuperJet2.push_back(miscJets[iii]);
					}
				}
				//std::cout << "Clustered jet masses: " << useSuperJet1.M() << " " << useSuperJet2.M() << std::endl;

				if( abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M()) < minMassDiff )
				{
					minMassDiff = abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M());
					bestSuperJet1 = _useSuperJet1;
					bestSuperJet2 = _useSuperJet2;
				}
			}
		}
	}



	////////////////////nmisc = 3/////////////////////////////////
	else if(nMiscJets == 3)
	{
		for(int aaa=0;aaa<2;aaa++)
		{
			for(int bbb=0;bbb<2;bbb++)
			{
				for(int ccc=0;ccc<2;ccc++)
				{
					if( (superJet1.size()==0)&&(superJet2.size() >0) )
					{
						if( (aaa==1) && (aaa==bbb) && (aaa==ccc) )continue; 
					}
					else if ( (superJet1.size() > 0)&&(superJet2.size() == 0)  )
					{
						if( (aaa==0) && (aaa==bbb) && (aaa==ccc) )continue;
					}
					else if ( (superJet1.size()==0)&&(superJet2.size() == 0) )
					{
						if((aaa==bbb) && (aaa==ccc)) continue;
					}
					useArray[0]=aaa;useArray[1]=bbb;useArray[2]=ccc;
					TLorentzVector useSuperJet1 = totSuperJet1;
					TLorentzVector useSuperJet2 = totSuperJet2;
					std::vector<TLorentzVector> _useSuperJet1 = superJet1;;
					std::vector<TLorentzVector> _useSuperJet2 = superJet2;;
					for(int iii = 0;iii<nMiscJets;iii++)
					{	
						if(useArray[iii] == 0)
						{
							useSuperJet1+=miscJets[iii];
							_useSuperJet1.push_back(miscJets[iii]);
						}
						else
						{ 
							useSuperJet2+=miscJets[iii];
							_useSuperJet2.push_back(miscJets[iii]);
						}
					}
			//std::cout << "Clustered jet masses: " << useSuperJet1.M() << " " << useSuperJet2.M() << std::endl;

					if( abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M()) < minMassDiff )
					{
							minMassDiff = abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M());
							bestSuperJet1 = _useSuperJet1;
							bestSuperJet2 = _useSuperJet2;
					} 
				}
			}
		}
	}
	
	else if(nMiscJets == 4)
	{
		for(int aaa=0;aaa<2;aaa++)
		{
			for(int bbb=0;bbb<2;bbb++)
			{
				for(int ccc=0;ccc<2;ccc++)
				{
					for(int ddd=0;ddd<2;ddd++)
					{
						if( (superJet1.size()==0)&&(superJet2.size() >0) )
						{
							if( (aaa==1) && (aaa==bbb) && (aaa==ccc) && (aaa==ddd) )continue; 
						}
						else if ( (superJet1.size() > 0)&&(superJet2.size() == 0)  )
						{
							if( (aaa==0) && (aaa==bbb) && (aaa==ccc) && (aaa==ddd) )continue;
						}
						else if ( (superJet1.size()==0)&&(superJet2.size() == 0) )
						{
							if((aaa==bbb) && (aaa==ccc) && (aaa==ddd)) continue;
						} 
						useArray[0]=aaa;useArray[1]=bbb;useArray[2]=ccc;useArray[3]=ddd;
						TLorentzVector useSuperJet1 = totSuperJet1;
						TLorentzVector useSuperJet2 = totSuperJet2;
						std::vector<TLorentzVector> _useSuperJet1 = superJet1;;
						std::vector<TLorentzVector> _useSuperJet2 = superJet2;;
						for(int iii = 0;iii<nMiscJets;iii++)
						{	
							if(useArray[iii] == 0)
							{
								useSuperJet1+=miscJets[iii];
								_useSuperJet1.push_back(miscJets[iii]);
							}
							else
							{ 
								useSuperJet2+=miscJets[iii];
								_useSuperJet2.push_back(miscJets[iii]);
							}
						}
				//std::cout << "Clustered jet masses: " << useSuperJet1.M() << " " << useSuperJet2.M() << std::endl;
						if( abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M()) < minMassDiff )
							{
								minMassDiff = abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M());
								bestSuperJet1 = _useSuperJet1;
								bestSuperJet2 = _useSuperJet2;
							} 
					}
				}
			}
		}
	}
	
	else if(nMiscJets == 5)
	{
		for(int aaa=0;aaa<2;aaa++)
		{
			for(int bbb=0;bbb<2;bbb++)
			{
				for(int ccc=0;ccc<2;ccc++)
				{
					for(int ddd=0;ddd<2;ddd++)
					{
						for(int eee=0;eee<2;eee++)
						{
							if( (superJet1.size()==0)&&(superJet2.size() >0) )
							{
								if( (aaa==1) && (aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) )continue; 
							}
							else if ( (superJet1.size() > 0)&&(superJet2.size() == 0)  )
							{
								if( (aaa==0) && (aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) )continue;
							}
							else if ( (superJet1.size()==0)&&(superJet2.size() == 0) )
							{
								if((aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee)) continue;
							} 
							useArray[0]=aaa;useArray[1]=bbb;useArray[2]=ccc;useArray[3]=ddd;useArray[4]=eee;
							TLorentzVector useSuperJet1 = totSuperJet1;
							TLorentzVector useSuperJet2 = totSuperJet2;
							std::vector<TLorentzVector> _useSuperJet1 = superJet1;;
							std::vector<TLorentzVector> _useSuperJet2 = superJet2;;
							for(int iii = 0;iii<nMiscJets;iii++)
							{	
								if(useArray[iii] == 0)
								{
									useSuperJet1+=miscJets[iii];
									_useSuperJet1.push_back(miscJets[iii]);
								}
								else
								{ 
									useSuperJet2+=miscJets[iii];
									_useSuperJet2.push_back(miscJets[iii]);
								}
							}
					//std::cout << "Clustered jet masses: " << useSuperJet1.M() << " " << useSuperJet2.M() << std::endl;
							if( abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M()) < minMassDiff )
								{
									minMassDiff = abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M());
									bestSuperJet1 = _useSuperJet1;
									bestSuperJet2 = _useSuperJet2;
								} 
						}
					}
				}
			}
		}		
	}
	
	else if(nMiscJets == 6)
	{
		for(int aaa=0;aaa<2;aaa++)
		{
			for(int bbb=0;bbb<2;bbb++)
			{
				for(int ccc=0;ccc<2;ccc++)
				{
					for(int ddd=0;ddd<2;ddd++)
					{
						for(int eee=0;eee<2;eee++)
						{
							for(int fff=0;fff<2;fff++)
							{
								if( (superJet1.size()==0)&&(superJet2.size() >0) )
								{
									if( (aaa==1) && (aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) && (aaa==fff) )continue; 
								}
								else if ( (superJet1.size() > 0)&&(superJet2.size() == 0)  )
								{
									if( (aaa==0) && (aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) && (aaa==fff) )continue;
								}
								else if ( (superJet1.size()==0)&&(superJet2.size() == 0) )
								{
									if((aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) && (aaa==fff)) continue;
								} 
								useArray[0]=aaa;useArray[1]=bbb;useArray[2]=ccc;useArray[3]=ddd;useArray[4]=eee; useArray[5]=fff;
								TLorentzVector useSuperJet1 = totSuperJet1;
								TLorentzVector useSuperJet2 = totSuperJet2;
								std::vector<TLorentzVector> _useSuperJet1 = superJet1;;
								std::vector<TLorentzVector> _useSuperJet2 = superJet2;;
								for(int iii = 0;iii<nMiscJets;iii++)
								{	
									if(useArray[iii] == 0)
									{
										useSuperJet1+=miscJets[iii];
										_useSuperJet1.push_back(miscJets[iii]);
									}
									else
									{ 
										useSuperJet2+=miscJets[iii];
										_useSuperJet2.push_back(miscJets[iii]);
									}
								}
						//std::cout << "Clustered jet masses: " << useSuperJet1.M() << " " << useSuperJet2.M() << std::endl;
								if( abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M()) < minMassDiff )
								{
										minMassDiff = abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M());
										bestSuperJet1 = _useSuperJet1;
										bestSuperJet2 = _useSuperJet2;
								}
							} 
						}
					}
				}
			}
		}
	}

	else if(nMiscJets == 7)
	{
		for(int aaa=0;aaa<2;aaa++)
		{
			for(int bbb=0;bbb<2;bbb++)
			{
				for(int ccc=0;ccc<2;ccc++)
				{
					for(int ddd=0;ddd<2;ddd++)
					{
						for(int eee=0;eee<2;eee++)
						{
							for(int fff=0;fff<2;fff++)
							{
								for(int ggg=0;ggg<2;ggg++)
								{
									if( (superJet1.size()==0)&&(superJet2.size() >0) )
									{
										if( (aaa==1) && (aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) && (aaa==fff) && (aaa==ggg) )continue; 
									}
									else if ( (superJet1.size() > 0)&&(superJet2.size() == 0)  )
									{
										if( (aaa==0) && (aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) && (aaa==fff) && (aaa==ggg) )continue;
									} 
									else if ( (superJet1.size()==0)&&(superJet2.size() == 0) )
									{
										if((aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) && (aaa==fff) && (aaa==ggg)) continue;
									}  									
									useArray[0]=aaa;useArray[1]=bbb;useArray[2]=ccc;useArray[3]=ddd;useArray[4]=eee, useArray[5] =fff, useArray[6]=ggg;
									TLorentzVector useSuperJet1 = totSuperJet1;
									TLorentzVector useSuperJet2 = totSuperJet2;
									std::vector<TLorentzVector> _useSuperJet1 = superJet1;;
									std::vector<TLorentzVector> _useSuperJet2 = superJet2;;
									for(int iii = 0;iii<nMiscJets;iii++)
									{	
										if(useArray[iii] == 0)
										{
											useSuperJet1+=miscJets[iii];
											_useSuperJet1.push_back(miscJets[iii]);
										}
										else
										{ 
											useSuperJet2+=miscJets[iii];
											_useSuperJet2.push_back(miscJets[iii]);
										}
									}
							//std::cout << "Clustered jet masses: " << useSuperJet1.M() << " " << useSuperJet2.M() << std::endl;
									if( abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M()) < minMassDiff )
									{
										minMassDiff = abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M());
										bestSuperJet1 = _useSuperJet1;
										bestSuperJet2 = _useSuperJet2;
									} 
								}
							}
						}
					}
				}
			}
		}		
	}


	else if(nMiscJets == 8)
	{
		for(int aaa=0;aaa<2;aaa++)
		{
			for(int bbb=0;bbb<2;bbb++)
			{
				for(int ccc=0;ccc<2;ccc++)
				{
					for(int ddd=0;ddd<2;ddd++)
					{
						for(int eee=0;eee<2;eee++)
						{
							for(int fff=0;fff<2;fff++)
							{
								for(int ggg=0;ggg<2;ggg++)
								{	
									for(int hhh=0;hhh<2;hhh++)
									{
										if( (superJet1.size()==0)&&(superJet2.size() >0) )
										{
											if( (aaa==1) && (aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) && (aaa==fff) && (aaa==ggg) && (aaa==hhh) )continue; 
										}
										else if ( (superJet1.size() > 0)&&(superJet2.size() == 0)  )
										{
											if( (aaa==0) && (aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) && (aaa==fff) && (aaa==ggg) && (aaa==hhh) )continue;
										}
										else if ( (superJet1.size()==0)&&(superJet2.size() == 0) )
										{
											if((aaa==bbb) && (aaa==ccc) && (aaa==ddd) && (aaa==eee) && (aaa==fff) && (aaa==ggg) && (aaa==hhh)) continue;
										}  
										useArray[0]=aaa;useArray[1]=bbb;useArray[2]=ccc;useArray[3]=ddd;useArray[4]=eee, useArray[5] =fff, useArray[6]=ggg; useArray[6]=hhh;
										TLorentzVector useSuperJet1 = totSuperJet1;
										TLorentzVector useSuperJet2 = totSuperJet2;
										std::vector<TLorentzVector> _useSuperJet1 = superJet1;;
										std::vector<TLorentzVector> _useSuperJet2 = superJet2;;
										for(int iii = 0;iii<nMiscJets;iii++)
										{	
											if(useArray[iii] == 0)
											{
												useSuperJet1+=miscJets[iii];
												_useSuperJet1.push_back(miscJets[iii]);
											}
											else
											{ 
												useSuperJet2+=miscJets[iii];
												_useSuperJet2.push_back(miscJets[iii]);
											}
										}
								//std::cout << "Clustered jet masses: " << useSuperJet1.M() << " " << useSuperJet2.M() << std::endl;
										if( abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M()) < minMassDiff )
										{
											minMassDiff = abs(useSuperJet1.M() - useSuperJet2.M())/min(useSuperJet1.M(),useSuperJet2.M());
											bestSuperJet1 = _useSuperJet1;
											bestSuperJet2 = _useSuperJet2;
										} 
									}
								}
							}
						}
					}
				}
			}
		}			
	}
	else{std::cout << "not enough cases for how many misc jets you want to consider .... ";}
	finalSuperJet1 = bestSuperJet1;
	finalSuperJet2 = bestSuperJet2;
	return;
}


