ls crab_projects/crab_BESTInputTrees_2015_* | grep crab_projects > resubmitCrab_All_2015.sh
sed -i 's/000:/000/g' resubmitCrab_All_2015.sh 
sed -i 's/crab_projects/crab resubmit --maxmemory 3500 -d crab_projects/g' resubmitCrab_All_2015.sh

ls crab_projects/crab_BESTInputTrees_2016_* | grep crab_projects > resubmitCrab_All_2016.sh
sed -i 's/000:/000/g' resubmitCrab_All_2016.sh 
sed -i 's/crab_projects/crab resubmit --maxmemory 3500 -d crab_projects/g' resubmitCrab_All_2016.sh

ls crab_projects/crab_BESTInputTrees_2017_* | grep crab_projects > resubmitCrab_All_2017.sh
sed -i 's/000:/000/g' resubmitCrab_All_2017.sh 
sed -i 's/crab_projects/crab resubmit --maxmemory 3500 -d crab_projects/g' resubmitCrab_All_2017.sh

ls crab_projects/crab_BESTInputTrees_2018_* | grep crab_projects > resubmitCrab_All_2018.sh
sed -i 's/000:/000/g' resubmitCrab_All_2018.sh 
sed -i 's/crab_projects/crab resubmit --maxmemory 3500 -d crab_projects/g' resubmitCrab_All_2018.sh
