# create submit files
ls crab_BEST_Producer_2015*QCD*.py > submitCrab_QCD_2015.sh
ls crab_BEST_Producer_2016*QCD*.py > submitCrab_QCD_2016.sh
ls crab_BEST_Producer_2017*QCD*.py > submitCrab_QCD_2017.sh
ls crab_BEST_Producer_2018*QCD*.py > submitCrab_QCD_2018.sh

sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_QCD_2015.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_QCD_2016.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_QCD_2017.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_QCD_2018.sh

ls crab_BEST_Producer_2015*Top*.py > submitCrab_Top_2015.sh
ls crab_BEST_Producer_2016*Top*.py > submitCrab_Top_2016.sh
ls crab_BEST_Producer_2017*Top*.py > submitCrab_Top_2017.sh
ls crab_BEST_Producer_2018*Top*.py > submitCrab_Top_2018.sh

sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_Top_2015.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_Top_2016.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_Top_2017.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_Top_2018.sh

ls crab_BEST_Producer_2015*Top*.py > submitCrab_Top_2015.sh
ls crab_BEST_Producer_2016*Top*.py > submitCrab_Top_2016.sh
ls crab_BEST_Producer_2017*Top*.py > submitCrab_Top_2017.sh
ls crab_BEST_Producer_2018*Top*.py > submitCrab_Top_2018.sh

sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_Top_2015.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_Top_2016.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_Top_2017.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_Top_2018.sh


ls crab_BEST_Producer_2015*_HT_*.py > submitCrab_HT_2015.sh
ls crab_BEST_Producer_2016*_HT_*.py > submitCrab_HT_2016.sh
ls crab_BEST_Producer_2017*_HT_*.py > submitCrab_HT_2017.sh
ls crab_BEST_Producer_2018*_HT_*.py > submitCrab_HT_2018.sh

sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_HT_2015.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_HT_2016.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_HT_2017.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_HT_2018.sh

ls crab_BEST_Producer_2015*_ZT_*.py > submitCrab_ZT_2015.sh
ls crab_BEST_Producer_2016*_ZT_*.py > submitCrab_ZT_2016.sh
ls crab_BEST_Producer_2017*_ZT_*.py > submitCrab_ZT_2017.sh
ls crab_BEST_Producer_2018*_ZT_*.py > submitCrab_ZT_2018.sh

sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_ZT_2015.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_ZT_2016.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_ZT_2017.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_ZT_2018.sh

ls crab_BEST_Producer_2015*_WB_*.py > submitCrab_WB_2015.sh
ls crab_BEST_Producer_2016*_WB_*.py > submitCrab_WB_2016.sh
ls crab_BEST_Producer_2017*_WB_*.py > submitCrab_WB_2017.sh
ls crab_BEST_Producer_2018*_WB_*.py > submitCrab_WB_2018.sh

sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_WB_2015.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_WB_2016.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_WB_2017.sh
sed -i 's/crab_BEST/crab submit -c crab_BEST/g' submitCrab_WB_2018.sh


# create combined sample submit files
echo "source submitCrab_QCD_2015.sh" > submitCrab_All_QCD.sh
echo "source submitCrab_QCD_2016.sh" >> submitCrab_All_QCD.sh
echo "source submitCrab_QCD_2017.sh" >> submitCrab_All_QCD.sh
echo "source submitCrab_QCD_2018.sh" >> submitCrab_All_QCD.sh

echo "source submitCrab_Top_2015.sh" > submitCrab_All_Top.sh
echo "source submitCrab_Top_2016.sh" >> submitCrab_All_Top.sh
echo "source submitCrab_Top_2017.sh" >> submitCrab_All_Top.sh
echo "source submitCrab_Top_2018.sh" >> submitCrab_All_Top.sh

echo "source submitCrab_HT_2015.sh" > submitCrab_All_HT.sh
echo "source submitCrab_HT_2016.sh" >> submitCrab_All_HT.sh
echo "source submitCrab_HT_2017.sh" >> submitCrab_All_HT.sh
echo "source submitCrab_HT_2018.sh" >> submitCrab_All_HT.sh

echo "source submitCrab_WB_2015.sh" > submitCrab_All_WB.sh
echo "source submitCrab_WB_2016.sh" >> submitCrab_All_WB.sh
echo "source submitCrab_WB_2017.sh" >> submitCrab_All_WB.sh
echo "source submitCrab_WB_2018.sh" >> submitCrab_All_WB.sh

echo "source submitCrab_ZT_2015.sh" > submitCrab_All_ZT.sh
echo "source submitCrab_ZT_2016.sh" >> submitCrab_All_ZT.sh
echo "source submitCrab_ZT_2017.sh" >> submitCrab_All_ZT.sh
echo "source submitCrab_ZT_2018.sh" >> submitCrab_All_ZT.sh

# create combined year submit files

#create combined submit files
echo "source submitCrab_QCD_2015.sh" > submitCrab_All_2015.sh
echo "source submitCrab_Top_2015.sh" >> submitCrab_All_2015.sh
echo "source submitCrab_HT_2015.sh" >> submitCrab_All_2015.sh
echo "source submitCrab_WB_2015.sh" >> submitCrab_All_2015.sh
echo "source submitCrab_ZT_2015.sh" >> submitCrab_All_2015.sh

echo "source submitCrab_QCD_2016.sh" > submitCrab_All_2016.sh
echo "source submitCrab_Top_2016.sh" >> submitCrab_All_2016.sh
echo "source submitCrab_HT_2016.sh" >> submitCrab_All_2016.sh
echo "source submitCrab_WB_2016.sh" >> submitCrab_All_2016.sh
echo "source submitCrab_ZT_2016.sh" >> submitCrab_All_2016.sh

echo "source submitCrab_QCD_2017.sh" > submitCrab_All_2017.sh
echo "source submitCrab_Top_2017.sh" >> submitCrab_All_2017.sh
echo "source submitCrab_HT_2017.sh" >> submitCrab_All_2017.sh
echo "source submitCrab_WB_2017.sh" >> submitCrab_All_2017.sh
echo "source submitCrab_ZT_2017.sh" >> submitCrab_All_2017.sh

echo "source submitCrab_QCD_2018.sh" > submitCrab_All_2018.sh
echo "source submitCrab_Top_2018.sh" >> submitCrab_All_2018.sh
echo "source submitCrab_HT_2018.sh" >> submitCrab_All_2018.sh
echo "source submitCrab_WB_2018.sh" >> submitCrab_All_2018.sh
echo "source submitCrab_ZT_2018.sh" >> submitCrab_All_2018.sh


