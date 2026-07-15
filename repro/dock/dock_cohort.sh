cd /workspace/MRDDA/dock
for L in daunorubicin idarubicin dactinomycin; do
  obabel $L.smi --gen3d -p 7.4 -O $L.pdbqt 2>/dev/null
  vina --receptor receptor.pdbqt --ligand $L.pdbqt     --center_x 33.03 --center_y 95.77 --center_z 51.57     --size_x 22 --size_y 22 --size_z 22 --exhaustiveness 16 --seed 42     --out ${L}_out.pdbqt > ${L}_vina.log 2>&1
  echo "DONE $L: $(grep "VINA RESULT" ${L}_out.pdbqt | head -1 | awk "{print \$4}") kcal/mol"
done
echo COHORT_DONE
