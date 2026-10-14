import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
if (!process.env.PRESENTATION_SKILL_DIR) throw new Error('Set PRESENTATION_SKILL_DIR to the installed presentation skill directory.');
const {finalizePresentation}=await import(path.join(process.env.PRESENTATION_SKILL_DIR,'container_tools/artifact_tool_utils.mjs'));
const root=process.cwd(),skill=process.env.PRESENTATION_SKILL_DIR;
const staging=path.join(root,process.env.STORY_STAGING_PATH??'output/review/model_demo_presentation_2026-10-14');
const story=JSON.parse(await fs.readFile(path.join(staging,'storyboard.json'),'utf8'));
const backup=path.join(root,'output/presentations/Case_Study_Retention_and_Revenue_Growth_2026-10-14.pptx');
const result=await finalizePresentation({
 workspaceDir:root,candidatePath:story.candidate,
 finalPath:path.join(root,process.env.STORY_FINAL_PATH??'output/presentations/Case_Study_Rebuilt.pptx'),
 pythonExecutable:process.env.RUNTIME_PYTHON || 'python3',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:story.slideCount,
 requiredNativeTableOwnerSlides:story.tableOwners,
 requiredNativeChartOwnerSlides:story.chartOwners,
 materializeLiteralChartWorkbooks:true,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
  ...story.tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 fontPolicy:{basis:'reference',families:['Calibri'],referencePath:backup,referenceSha256:createHash('sha256').update(await fs.readFile(backup)).digest('hex')},
 tableArithmeticContracts:[{slide:story.slides.find(r=>r.title==='Subscription value reconciliation').slide,table:1,label_column:0,total_row:11,value_columns:[1],component_rows:[1,2,3,4,5,6,7,8,9,10]}],
 verifyArtifactToolImport:true,
 receiptPath:path.join(staging,path.basename(process.env.STORY_FINAL_PATH??'Case_Study_Rebuilt.pptx')+'.validation.json')
});
console.log(JSON.stringify(result));
