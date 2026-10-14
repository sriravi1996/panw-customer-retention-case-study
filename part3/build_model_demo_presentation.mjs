/** Plain-language case-study deck. All empirical figures bind to current outputs. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
const {Presentation,PresentationFile}=await import(process.env.ARTIFACT_TOOL_ENTRY || '@oai/artifact-tool');
if (!process.env.PRESENTATION_SKILL_DIR) throw new Error('Set PRESENTATION_SKILL_DIR to the installed presentation skill directory.');
const {applyPresentationChartFont}=await import(path.join(process.env.PRESENTATION_SKILL_DIR,'container_tools/artifact_tool_utils.mjs'));
const root=process.cwd(), staging=path.join(root,process.env.STORY_STAGING_PATH??'output/review/model_demo_presentation_2026-10-14');
const read=async f=>JSON.parse(await fs.readFile(path.join(root,f),'utf8'));
const b=await read(process.env.STORY_BUNDLE_PATH??'part3/model_demo_presentation_data.json');
const extra=b.extra;
const sel=await read('part2/outputs/model_selection_record.json');
const metrics=await read('part2/outputs/metrics.json');
const selected=sel.selected_model.experiment+'_'+sel.selected_model.model_type;
const finalKey=sel.refit.artifact.replace(/\.pkl$/,'');
const ft=metrics[finalKey].final_test, ms=metrics[selected].model_selection;
const hgb=sel.selected_model.model_type==='hgb';
const C={navy:'#0D1B2A',orange:'#FF5A1F',teal:'#008C95',green:'#1A7C51',ink:'#23313D',muted:'#526271',light:'#F1F4F6',line:'#D3DCE2',white:'#FFFFFF'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const story=[],chartOwners=[],tableOwners=[];
const clean=t=>String(t).replace(/[\u2014\u2013]/g,', ').replace(/\s+,/g,',');
const fmt=n=>Number(n).toLocaleString('en-US'), ap=n=>Number(n).toFixed(4);
function rect(s,x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function txt(s,t,x,y,w,h,size=24,color=C.ink,bold=false){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=clean(t);q.text.insets=0;q.text.style={typeface:'Calibri',fontSize:size,color,bold};return q;
}
function slide(title,notes,source='',section=''){
 const s=p.slides.add(),n=p.slides.items.length;s.background.fill=C.white;
 rect(s,0,0,1280,90,C.navy);rect(s,0,90,1280,4,C.orange);
 txt(s,title,56,23,1135,54,34,C.white,true);txt(s,n,1211,24,40,40,21,C.orange,true);
 s.speakerNotes.text=clean(notes+'\nSource: '+source);
 story.push({slide:n,title,section,source});return s;
}
function foot(s,t){txt(s,t,56,656,1168,48,17,C.muted);}
function body(s,t,y=126,h=100,size=25){return txt(s,t,56,y,1168,h,size);}
function label(s,t,x,y,w=400,color=C.teal){txt(s,t,x,y,w,45,27,color,true);}
function table(s,rows,x=56,y=160,w=1168,h=360,widths=null,font=22){
 const t=s.tables.add({rows:rows.length,columns:rows[0].length,left:x,top:y,width:w,height:h,values:rows.map(r=>r.map(clean)),columnWidths:widths??rows[0].map(()=>w/rows[0].length)});
 t.cells.block({row:0,column:0,rowCount:rows.length,columnCount:rows[0].length}).assign({fill:C.white,textStyle:{typeface:'Calibri',fontSize:font,color:C.ink},margins:{left:12,right:12,top:rows.length>=8?4:8,bottom:rows.length>=8?4:8}});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:rows[0].length}).assign({fill:C.navy,textStyle:{typeface:'Calibri',fontSize:font,bold:true,color:C.white}});
 t.borders.assign({fill:C.line,width:0.6,style:'solid'});
 for(let i=2;i<rows.length;i+=2)t.cells.block({row:i,column:0,rowCount:1,columnCount:rows[0].length}).assign({fill:C.light});
 tableOwners.push(p.slides.items.length);return t;
}
function chart(s,type,cfg){
 cfg.series=cfg.series.map(r=>({...r,values:r.values?.map(v=>Number(Number(v).toFixed(10)))}));
 const c=s.charts.add(type,cfg);applyPresentationChartFont(c,{fontFamily:'Calibri'});chartOwners.push(p.slides.items.length);return c;}
function bars(s,title,categories,values,x=56,y=150,w=750,h=425,format='0.0%',color=C.teal,horizontal=true){
 return chart(s,'bar',{position:{left:x,top:y,width:w,height:h},title,titleTextStyle:{fontSize:22,fill:C.ink},categories:horizontal?[...categories].reverse():categories,series:[{name:title,values:horizontal?[...values].reverse():values,fill:color}],hasLegend:false,barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:75},
 xAxis:{title:horizontal?'':'Ranking method',numberFormatCode:'General',textStyle:{fontSize:16,fill:C.muted}},
 yAxis:{title:horizontal?'':title,numberFormatCode:format,textStyle:{fontSize:17,fill:C.ink},min:0,majorUnit:horizontal?(format==='0.000'?(Math.max(...values)>.05?.05:.005):format==='0.0%'?.02:format==='#,##0'?5000:undefined):undefined},
 dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:18,fill:C.ink,bold:true}},chartFill:C.white,plotAreaFill:C.white});
}
function step(s,n,title,detail,x,y,w=330){txt(s,String(n),x,y,55,55,39,C.orange,true);txt(s,title,x+65,y+1,w-65,62,25,C.ink,true);txt(s,detail,x+65,y+74,w-65,150,23);}

// 1. Minimal cover.
{
 const s=slide('B2B cybersecurity case study',
 'Open with the business problem: customer-success teams have limited time to review renewals. I use synthetic cybersecurity subscription histories to test whether customer-experience signals can identify cancellations and help decide whom to contact. The presentation distinguishes cancellation identification, financial exposure and actual intervention impact. All prices, company histories and outcomes are simulated. Target 1 minute.',
 'Case-study brief; customer_management_scope.md','Opening');
 s.background.fill=C.navy;txt(s,'Customer Retention\nand Revenue Growth',70,180,1100,155,54,C.white,true);
 txt(s,'A B2B Cybersecurity Case Study Using Synthetic Data',70,390,1100,60,29,C.orange);
 txt(s,'10,000 purchasing accounts\nFive years of operating history',70,495,1100,95,26,C.white);
}
// 2. Definition first, as requested in the meeting.
{
 const s=slide('What cancellation means in this study',
 'A prediction describes one subscription product for one purchasing account and one contract term. The positive outcome requires an actual cancellation event. A mature zero means no such event was recorded, and does not establish renewal. A company may cancel one product and keep others. At the prediction date, I use only available history. Ninety days later the contract expires. I wait a further 90 days to observe the outcome. This follow-up assumption is separate from any product entitlement grace period. The meeting accepted multi-year terms if I stated these definitions. Target 2 minutes.',
 'part2/src/build_features.py; transcript 02:34 to 06:17','Business problem');
 body(s,'Predict a recorded product-subscription cancellation. A whole company can remain a customer.',125,100,28);
 step(s,1,'Prediction date','90 days before the contract expires',70,280,350);
 step(s,2,'Contract expiry','The current subscription term ends',475,280,350);
 step(s,3,'Outcome observed','Follow cancellation events through expiry + 90 days',880,280,330);
 txt(s,'One row = one subscription contract at its prediction date',70,560,1140,55,27,C.teal,true);
 foot(s,'Lower annual value is contraction. No recorded cancellation does not prove a successful renewal.');
}
// 3. Hierarchy and decision unit.
{
 const s=slide('One company can have several renewal decisions',
 'Parent organizations group related purchasing accounts. Purchasing accounts make commercial purchases. Each account can hold multiple product-subscription lines, with renewal terms recorded separately. Teams use products through deployments, which are operating installations. Teams are not independent cancellation labels in this dataset. We predict at contract level and select one lead contract per account when assigning reviews. A lead is the contract with the highest score under the chosen policy. The same account can therefore have a different lead under a different ranking. All accounts are assumed directly managed for the case study. Target 2 minutes.',
 'data/raw/parent_orgs.csv; companies.csv; order_item_fact.csv; customer_management_scope.md; transcript 02:01 to 04:56','Business problem');
 table(s,[['Level','Plain meaning','Used for'],['Parent organization',fmt(b.data_quality.n_parent_orgs)+' groups of related accounts','Grouping related customers'],['Purchasing account',fmt(b.data_quality.n_accounts)+' buying accounts','Assigning customer reviews'],['Subscription contract','One product line for one term','Predicting cancellation'],['Team and deployment','People using an installation','Measuring adoption and product value']],56,140,1168,360,[305,490,373],23);
 body(s,'Review capacity assumption: 30 purchasing accounts per cycle',545,60,28);
 foot(s,'The customer team manages all accounts in this scenario. Product use and commercial cancellation remain different events.');
}
// 4. Generation as a chronological process.
{
 const s=slide('How I created the synthetic customer histories',
 'No customer CSVs were supplied, so I generated connected B2B cybersecurity records. Random draws create variation in customer size, initial product ownership, contract terms and quantities. Hidden customer-experience conditions influence both observable product behavior and renewal risk, with noise and independent outcome draws. Renewals run in calendar order across products so a future cancellation cannot change a past portfolio. Telemetry, support and relationship records follow valid contract and deployment lifetimes. Finally I inject declared source errors to exercise cleaning. The simulator truth labels and risk probabilities never enter model inputs. Company sizes and terms use categorical draws. Budgets are lognormal. Ticket counts use a Poisson process scaled by exposure and onboarding. Successful workflows use binomial draws and cancellation is sampled as a Bernoulli outcome. Gaussian noise varies the shared experience measurements. These distribution choices are assumptions. A fixed seed makes the process reproducible. The observations imitate plausible mechanisms rather than representing PANW measurements. Target 2 minutes.',
 'src/generate_data.py; config/simulation.yaml; transcript 00:31 to 01:43','Data foundation');
 table(s,[['Step','What is generated','Why it matters'],['1. Buying accounts','Company size, product choices, quantities and term lengths','Different commercial histories'],['2. Customer experience','Deployment progress, product use and service problems','Plausible cancellation mechanisms'],['3. Renewals over time','Purchases, renewals, contractions and cancellations','Chronological commercial outcomes'],['4. Operating records','Weekly product signals, support cases and interactions','Observable history for prediction'],['5. Source errors','Duplicates, invalid dates and broken references','A realistic cleaning exercise']],56,140,1168,415,[250,520,398],23);
 foot(s,'Primary seed 137. Experience generates noisy operating signals and stochastic outcomes. Hidden probabilities never enter the model.');
}
// 5. State major assumptions before findings.
{
 const s=slide('The assumptions behind the simulated outcomes',
 'These annual cancellation risk anchors are assumptions: 8% for small and medium businesses, 5% midmarket, 2.5% enterprise and 1% large enterprise. They do not mean size itself causes churn. Experience and commercial conditions adjust individual risk. The simulator converts annual risk to the contract term, so unchanged 8% annual risk becomes 1 minus 0.92 cubed, about 22.13%, over three years. Realized contract cancellation percentages need not equal annual anchors. Main instrumentation is reliable. Missing or inapplicable product histories remain distinguishable from confirmed zero use. The observation period is October 2021 to October 2026, with opening histories back to October 2018. Prices and customer mix are not calibrated to PANW financials. No intervention effect is simulated. Target 2 minutes.',
 'config/simulation.yaml; docs/simulation_assumptions.md; transcript 05:01 to 09:25','Data foundation');
 bars(s,'Assumed annual cancellation risk',['Small and medium business','Midmarket','Enterprise','Large enterprise'],[.08,.05,.025,.01],56,145,690,400);
 label(s,'Additional assumptions',815,150,400);
 txt(s,'Contract terms: 1 to 3 years\n\nReliable operating measurements\n\nExperience can change renewal risk\n\nSynthetic prices and customer mix',815,215,390,320,24);
 foot(s,'These are simulation choices. Example: 8% annual risk becomes 22.13% over three years before experience adjustments.');
}
// Declared mechanisms precede all model evidence.
{
 const s=slide('Assumed combinations that raise cancellation risk',
 'The simulation deliberately includes interactions: two problems together can raise risk more than the sum of separate warning signs. These are chosen mechanisms, not empirical findings about Palo Alto Networks. Deployment gap above 45 percent with value deterioration above 35 percent adds 3.2 to log odds. Value deterioration above 55 percent with deployment gap above 15 percent adds 2.2. Commercial pressure above 60 percent with value deterioration above 20 percent adds 1.2. Service problems above 20 percent with deployment gap above 30 percent add 0.8. Smooth main effects also contribute. A separate fixed reference population of 20,000 latent customers per segment sets intercepts to the annual risk anchors without using outcome labels. Term conversion and caps follow afterward. Coverage blends 75 percent shared experience with 25 percent deployment trajectory and weekly noise. Workflow success has mean 0.90 minus 0.80 times value deterioration, plus noise. A probabilistic draw determines cancellation. The learned model receives noisy observed history, not those latent states or the generator formula. These strengths were fixed before generating the reported populations. Explain the mechanism, not every coefficient.',
 'config/simulation.yaml experience_scenario; src/generate_data.py; simulation_assumptions.md SA-011','Simulation assumptions');
 table(s,[['Combined experience','What we could observe'],['Poor deployment and weak product value','Little intended scope deployed and unsuccessful workflows'],['Value deterioration with incomplete deployment','Poor workflow success alongside a deployment gap'],['Commercial pressure and weak product value','An unfavorable quote or budget concern plus weak adoption'],['Service problems and poor deployment','Unresolved support impact alongside incomplete protection']],56,145,1168,325,[530,638],24);
 txt(s,'Assumed experience creates both operating signals and cancellation risk.',56,510,1168,70,29,C.teal,true);
 foot(s,'These relationships and their strengths are simulation choices. Healthy customers can cancel and struggling customers can renew.');
}
// 6. Connected datasets and cleaning.
{
 const s=slide('Connected data supports a usable renewal record',
 'The four required relational tables are users, orders, order_items and products. Users are business contacts rather than the unit of retention. Order items link product quantities and values to orders. Additional records describe company hierarchy, teams, deployments, entitlements, operating observations, support cases, CRM interactions and renewal quotes. A foreign-key check means that an identifier must point to a real record. A join-cardinality check prevents combining tables from multiplying transactions. Recoverable duplicates keep their valid dated copy. Irrecoverable rows go to quarantine with reasons. The final renewal feature row contains only facts available by its prediction date. Part 1 passed 35 checks on the saved data. Target 2 minutes.',
 'outputs/quality_report.json; src/ingest_clean.py; src/integrate.py','Data foundation');
 table(s,[['Source','What it tells us','Cleaned scale'],['Users, orders, order items, products','Who bought which product and for how much',fmt(b.data_quality.n_orders_clean)+' orders\n'+fmt(b.data_quality.n_order_items_clean)+' order items'],['Deployments and weekly observations','What is installed and how it performs',fmt(b.data_quality.n_deployments)+' deployments\n'+(b.data_quality.n_weekly_obs/1e6).toFixed(2)+' million weekly records'],['Support cases and interactions','Service problems and relationship history',fmt(b.data_quality.n_support_tickets)+' support cases'],['Renewal quotes and contract events','Commercial concerns and observed outcomes','Joined by account and contract']],56,140,1168,360,[420,440,308],22);
 body(s,'Cleaning removed '+fmt(b.data_quality.n_duplicate_orders_removed)+' duplicate orders and quarantined '+fmt(b.data_quality.n_invalid_date_quarantined)+' orders with invalid dates.',545,82,25);
 foot(s,'35 Part 1 checks passed. Deployment count covers the full generated history, rather than active installations on one date.');
}
// 7. Mandatory purchasing EDA.
{
 const s=slide('Purchasing patterns and average order value',
 'EDA means exploring the data before modeling. The chart counts distinct eligible orders containing each product, not incompatible license quantities. Support purchases can accompany product purchases and renewals. Average order value is total eligible order value divided by eligible order count. The highest AOV account is '+b.eda.highest_aov.company+', at '+b.eda.highest_aov.aov_m+' million across '+b.eda.highest_aov.order_count+' order. Its largest product contribution is '+b.eda.highest_aov.top_product+'. The highest AOV repeat buyer is '+b.eda.repeat_buyer.company+', at '+b.eda.repeat_buyer.aov_m+' million across '+b.eda.repeat_buyer.orders+' orders. These large amounts arise from invented prices, quantities and terms. They are not PANW pricing estimates or evidence of customer loyalty.',
 'outputs/eda_top20_ordered_items.csv; outputs/eda_summary.json','EDA');
 bars(s,'Distinct orders containing each product',b.eda.top5_products.map(x=>x.name),b.eda.top5_products.map(x=>x.orders),45,140,745,430,'#,##0');
 label(s,'Largest average order value',825,155,370);
 txt(s,b.eda.highest_aov.company+'\n$'+b.eda.highest_aov.aov_m+'M across '+b.eda.highest_aov.order_count+' order',825,225,385,110,28,C.orange,true);
 txt(s,'Largest product contribution\nwithin this customer’s purchase:\n'+b.eda.highest_aov.top_product,825,345,385,110,23);
 txt(s,'Repeat buyer: '+b.eda.repeat_buyer.company+'\n$'+b.eda.repeat_buyer.aov_m+'M across '+b.eda.repeat_buyer.orders+' orders',825,475,385,100,24);
 foot(s,'Average order value = total eligible order value ÷ number of eligible orders. A large purchase alone does not demonstrate loyalty.');
}
// 9. Explain behavior with an explicitly illustrative native chart.
{
 const s=slide('Healthy operation versus declining adoption',
 'Cybersecurity products can protect customers without frequent administrator logins. Coverage measures the share of intended scope actually protected, for example protected endpoints out of eligible endpoints. Productive use means that the relevant workflow succeeds, such as successful scans or automated response. For an endpoint example the intended and eligible pools can be the same. In the feature table, average coverage specifically uses active divided by eligible deployment units, while deployment reach compares deployed targets with the intended environment, which can be broader. Neither ratio is a measured percentage of attacks prevented. I consider both level and change over time. Stable high protection can be healthy with little manual activity. Stable low coverage may mean persistent under-adoption. Falling coverage is a different signal. Recovery may follow resolved issues. The curves are illustrative explanations, not measured estimates or model thresholds. Missing required measurements need investigation rather than automatically meaning customer disengagement. Target 2 minutes.',
 'src/generate_data.py; part2/src/build_features.py; transcript 08:31 to 09:25','Customer behavior');
 chart(s,'line',{position:{left:45,top:145,width:760,height:430},title:'Illustrative protection patterns',categories:['Earlier','Middle','Recent'],series:[
 {name:'Healthy operation',values:[.85,.86,.85],line:{fill:C.green,width:3}},
 {name:'Persistent under-adoption',values:[.25,.26,.24],line:{fill:'#9A6A15',width:3}},
 {name:'Declining adoption',values:[.80,.55,.28],line:{fill:C.orange,width:3}},
 {name:'Recovery',values:[.25,.50,.76],line:{fill:C.teal,width:3}}],
 legend:{position:'bottom',overlay:false,textStyle:{fontSize:17}},xAxis:{title:'Time before prediction'},yAxis:{title:'Percentage of intended scope covered',min:0,max:1,numberFormatCode:'0%'},chartFill:C.white});
 label(s,'Coverage in plain language',840,160,370);
 txt(s,'800 protected endpoints\n÷ 1,000 intended endpoints\n= 80% coverage',840,225,365,150,28,C.teal,true);
 txt(s,'Activity is useful only when it reflects protection or product value.',840,425,365,130,26);
 foot(s,'The curves and endpoint example illustrate concepts. They are not calculated findings from the dataset.');
}
// Complete selected feature inventory, grouped across three readable slides.
const featureGroups={
 'Deployment':[
 ['deployment_reach_30d','Share of intended scope deployed'],
 ['onboarding_delay_days','Days the initial-value milestone is overdue'],
 ['deployment_reach_30d_missing','Deployment reach unavailable (indicator)']],
 'Product use and operating results':[
 ['workflow_success_30d','Share of attempted workflows that succeed'],
 ['productive_use_delta_30d','Recent change in workflow success'],
 ['avg_coverage_90d','Average coverage of eligible deployment units'],
 ['coverage_trend_90d','Change in coverage over time'],
 ['obs_completeness_90d','Share of expected observations received'],
 ['policy_events_per_unit_90d','Policy activity per eligible unit'],
 ['remediation_velocity_90d','Speed of fixing recorded issues'],
 ['automation_adoption_90d','Share of relevant work using automation'],
 ['scan_success_rate_90d','Share of product scans that succeed'],
 ['workflow_success_30d_missing','Workflow success unavailable (indicator)']],
 'Support experience':[
 ['unresolved_customer_impact_hours_90d','Hours of unresolved customer impact'],
 ['repeat_sla_breaches_90d','Open cases breaching the service target'],
 ['p1_tickets_90d','Critical support tickets'],
 ['open_tickets_90d','Cases still open at prediction'],
 ['avg_open_age_hrs_90d','Average age of open cases'],
 ['avg_resolution_hrs_90d','Average time to resolve completed cases'],
 ['avg_resolution_hrs_90d_missing','Resolution time unavailable (indicator)'],
 ['tickets_90d','Total recent support tickets']],
 'Customer relationship':[
 ['days_without_confirmed_owner','Days since a recorded owner interaction'],
 ['overdue_agreed_actions','Agreed follow-up actions overdue'],
 ['executive_interactions_180d','Recorded executive interactions'],
 ['marketing_engagements_180d','Recorded marketing and event engagements'],
 ['days_since_last_interaction','Days since the last recorded interaction'],
 ['days_since_last_interaction_missing','Interaction recency unavailable (indicator)']],
 'Commercial and renewal context':[
 ['quoted_unit_price_change_pct','Quoted unit-price change'],
 ['dated_budget_concern','A recorded budget concern'],
 ['days_to_renewal','Days until expiry (fixed at 90 in this design)'],
 ['tenure_days','Age of the purchasing relationship'],
 ['employee_count','Customer employee count'],
 ['acv','Annual value of this contract'],
 ['total_acv_in_window','Annual value in the renewal window'],
 ['median_prior_renewal_delay_days','Immediate prior subscription transition gap'],
 ['median_prior_renewal_delay_missing','Prior transition gap unavailable (indicator)']],
 'Product portfolio':[
 ['n_products_in_window','Product count in the renewal window'],
 ['n_families_in_window','Product-family count in the renewal window'],
 ['n_active_products_in_family','Active products in the same family'],
 ['family_sibling_cancellations_1yr','Other family-product cancellations in the past year'],
 ['family_adoption_completeness','Share of the available family catalog owned']],
 'Category inputs':[
 ['product_family','Product family'],['size_segment','Customer-size group'],
 ['industry','Customer industry'],['region','Customer region']]
};
const presentedFeatures=Object.values(featureGroups).flat().map(r=>r[0]);
const expectedFeatures=[...sel.selected_model.num_features,...sel.selected_model.cat_features];
if(new Set(presentedFeatures).size!==expectedFeatures.length || expectedFeatures.some(f=>!presentedFeatures.includes(f)))throw new Error('Feature slides do not cover the selected schema exactly');
function featureColumn(s,name,x,y,w=550){
 const rows=featureGroups[name];label(s,name+' ('+rows.length+')',x,y,w);
 rows.forEach((r,i)=>txt(s,r[1],x,y+62+i*33,w,34,23));
 s.speakerNotes.text+='\n'+name+':\n'+rows.map(r=>r[0]+' = '+r[1]).join('\n');
}
{
 const s=slide('Model inputs: deployment and product performance',
 'A feature is a measurable fact supplied to a model. These slides list every selected input, not merely examples. There are '+expectedFeatures.length+' total inputs: '+sel.selected_model.num_features.length+' numeric and '+sel.selected_model.cat_features.length+' categorical. Deployment reach and workflow success use 30 days; other operating measures generally use 90 days. Missing indicators are yes/no fields showing that an input was unavailable. They preserve the distinction between unknown and confirmed zero. Average coverage uses active units divided by eligible deployment units, averaged across weekly observations. Deployment reach uses deployed targets divided by the intended target environment. For infrastructure products, intended scope can be larger than the eligible capacity. A scan measure only applies to products that perform scans. No future commercial outcome or simulator truth is eligible. Explain the groups rather than reading every field. Target 1.5 minutes.',
 'part2/src/_features.py; model_selection_record.json','Model inputs');
 txt(s,'Complete input set: '+expectedFeatures.length+' fields, including '+sel.selected_model.num_features.length+' numeric and '+sel.selected_model.cat_features.length+' categorical inputs',56,108,1168,28,21,C.muted);
 featureColumn(s,'Deployment',56,150);featureColumn(s,'Product use and operating results',670,150);
 txt(s,'Deployment reach: deployed targets ÷ intended targets.\nCoverage: active units ÷ eligible deployment units.',56,380,520,145,26,C.teal,true);
 foot(s,'Recent deployment and workflow measures use 30 days. Other operating signals generally use 90 days. All inputs precede prediction.');
}
{
 const s=slide('Model inputs: support and customer relationships',
 'The support group records unresolved problems, their severity and how long they persist. A service-level agreement, or SLA, is a promised service target. The breached-case feature counts open cases that have breached that target at prediction, rather than every historical breach. P1 means a critical support ticket. Relationship inputs capture executive interactions, marketing activity, recorded follow-up and recency. They are candidate signals, not proof that additional marketing would prevent cancellation. The owner-bearing interaction field is a proxy for relationship recency, not proof that nobody owns the account. Explain the groups rather than reading every field. Target 1.5 minutes.',
 'part2/src/_features.py; build_features.py; transcript 07:28 to 08:30','Model inputs');
 featureColumn(s,'Support experience',56,140);featureColumn(s,'Customer relationship',670,140);
 foot(s,'Support history: 90 days. Relationship counts: 180 days. Recency fields use the latest known qualifying interaction.');
}
{
 const s=slide('Model inputs: commercial and portfolio context',
 'Annual contract value expresses a subscription value per year. Portfolio features summarize what is owned or renewing. Family catalog ownership is not evidence that every product is well deployed. The prior transition-gap code name says median, but the implementation currently measures the immediate previous transition gap, not a median across the full history. Days to renewal is always 90 in this design and does not add variation. Company size and tenure are contextual, not causes or intervention targets by themselves. Four categorical inputs describe product family, customer size, industry and region. Explain the groups rather than reading every field. Target 1.5 minutes.',
 'part2/src/_features.py; build_features.py; model_selection_record.json','Model inputs');
 featureColumn(s,'Commercial and renewal context',56,135);featureColumn(s,'Product portfolio',670,135);
 featureColumn(s,'Category inputs',670,405);
 foot(s,'Context is measured at prediction. The prior transition-gap field uses one predecessor. Days until expiry is fixed at 90 in this design.');
}
// 11. Method and temporal procedure.
{
 const s=slide('Training and evaluation over time',
 'The approaches are a constant prevalence reference, a fixed health score, regularized logistic regression and histogram gradient boosting. The learned models use three feature sets: experience signals, size/value context, and full context. Historic model-selection average precision chooses the model. Class-weighted and unweighted candidates are tested without oversampling. The selected HGB uses no class weights, 100 trees and depth three. Refit combines training and selection only, yielding '+fmt(b.splits.refit_n)+' rows and '+fmt(b.splits.refit_pos)+' cancellations. Later labels never enter fitting. Prediction occurs 90 days before expiry and label maturity 90 days after expiry, so training needs a 180-day maturity lag. Boundary rows are purged to ensure each fitting date only uses known outcomes. The reported synthetic populations are generated after the scenario assumptions are fixed. Real customer performance remains unknown.',
 'part2/src/train.py; part2/src/make_splits.py; model_selection_record.json; transcript 10:22 to 10:50 and 13:23 to 13:45','Modeling');
 table(s,[['Stage','Prediction period','Rows / cancellations'],['Train candidates','Through 4 April 2023',fmt(b.splits.train.n)+' / '+b.splits.train.pos],['Choose model','October 2023 to April 2024',fmt(b.splits.ms.n)+' / '+b.splits.ms.pos],['Check later period','October 2024 to April 2025',fmt(b.splits.cal.n)+' / '+b.splits.cal.pos],['Later evaluation','October 2025 to April 2026',fmt(b.splits.final_test.n)+' / '+b.splits.final_test.pos]],56,140,1168,300,[310,530,328],23);
 txt(s,'Candidates: health rule, logistic regression and boosted decision trees',56,480,1168,56,26,C.teal,true);
 txt(s,'Refit uses '+fmt(b.splits.refit_n)+' earlier observations.\n'+fmt(b.splits.boundary_purged)+' windows are excluded near fitting dates. '+fmt(b.splits.scoring.n)+' await outcomes.',56,555,1168,70,24);
 foot(s,'The modeling table contains '+fmt(extra.retained_labeled+b.splits.boundary_purged+b.splits.scoring.n)+' eligible subscription windows. Class weights were tested. No oversampling. Synthetic evaluation under fixed assumptions.');
}
// Ranking quality and the business metric serve distinct purposes.
{
 const s=slide('The model learns useful ranking patterns',
 'Average precision summarizes precision as the ranked list recovers more cancellations. The left chart is the historical model-selection period and determines the selected model. The selected boosted-tree model has AP '+ap(ms.ap)+'. The right values describe the refitted model on a later synthetic period. Later AP is '+ap(ft.ap)+' versus '+ap(b.ft_aps.rule)+' for health rules. Later prevalence is '+b.ft_prevalence+' percent, with '+b.splits.final_test.pos+' cancellations among '+fmt(b.splits.final_test.n)+' contracts. AP does not guarantee superiority at every cutoff. Precision among the top 30 account reviews is therefore checked separately. The scenario intentionally encodes experience interactions. A successful model demonstrates recovery of those assumptions, not independent proof of actual customer drivers.',
 'part2/outputs/metrics.json; model_selection_record.json','Modeling');
 bars(s,'Average precision in model selection',['Prevalence reference','Health rule','Logistic regression','Boosted trees'],[b.ms_aps.prevalence,b.ms_aps.rule,b.ms_aps.lr,b.ms_aps.hgb],50,145,755,425,'0.000',C.teal,false);
 label(s,'Later synthetic evaluation',855,155,360);
 txt(s,'Average precision\nModel '+ap(ft.ap)+'\nRules '+ap(b.ft_aps.rule),855,230,350,180,31,C.orange,true);
 txt(s,'ROC-AUC: '+ft.roc_auc.toFixed(3)+'\nOverall separation of outcomes',855,445,350,100,24);
 foot(s,'Model selection uses historical AP. The business comparison uses precision at 30 account reviews.');
}

// Importance is descriptive model reliance within declared mechanisms.
{
 const s=slide('Successful use and coverage lead model importance',
 'Permutation importance measures the fall in selection-set average precision when one input is shuffled. Workflow success and average coverage dominate in this scenario. Low successful use and poor protection jointly represent unrealized product value in the declared generator. Employee count provides context for the assumed segment risk anchors and is not a causal reason to cancel. The 30-day change in workflow success describes recovery or deterioration. Deployment reach overlaps with average coverage. The smaller inputs have greater uncertainty relative to their importance, so do not present five equally robust causal drivers. Importance does not itself establish effect direction. The large top-two importances are expected because the simulator deliberately links these experiences to cancellation.',
 'part2/outputs/feature_importance.csv; config/simulation.yaml','Model interpretation');
 const names={workflow_success_30d:'Workflow success in recent 30 days',avg_coverage_90d:'Average protection coverage',employee_count:'Customer employee count',productive_use_delta_30d:'Change in workflow success',deployment_reach_30d:'Recent deployment reach'};
 bars(s,'AP decrease when input is shuffled',b.top5_fi.map(x=>names[x.feature]??x.plain),b.top5_fi.map(x=>x.ap_drop),45,150,790,410,'0.000');
 txt(s,'Strongest clues\nSuccessful workflows and protection coverage.\n\nContext\nSize affects the assumed baseline risk.',870,180,330,300,25);
 foot(s,'These are model reliance measures within the simulation. Correlated features share information. Smaller importances are less stable.');
}

// 14. The missing score calculation and risk times value explanation.
{
 const s=slide('Risk score and value-based priority',
 'Customer history enters the selected trained model. It combines learned patterns from 100 shallow trees and returns a score between zero and one. Training learns those tree splits and values from historic labels. I do not manually weight the top five features or copy the simulator cancellation formula into the ML model. The score orders contracts by modeled cancellation tendency. Because it is uncalibrated, 0.4 should not be presented as a validated 40 percent chance. ACV means annual contract value: a 300-thousand-dollar subscription over three years has ACV of 100 thousand dollars, assuming equal annual value. Multiplying score by ACV creates a priority index. In the illustrative example, A has higher modeled risk, while B has higher value-weighted priority. Rank contracts using the chosen policy, then keep one lead contract per purchasing account. This heuristic is useful to test; it is not predicted revenue loss or savings. Target 3 minutes.',
 'part2/src/train.py; part2/src/build_review_cohort.py; transcript 11:49 to 12:21','Risk to action');
 txt(s,'Customer history',56,132,350,45,29,C.teal,true);txt(s,'Trained decision trees',465,132,390,45,29,C.teal,true);txt(s,'Risk score: 0 to 1',900,132,330,45,29,C.teal,true);
 txt(s,'Deployment, use, service\nand commercial context',56,190,340,85,24);txt(s,'Each tree adjusts a starting score.\nThe total becomes a 0 to 1 score.',465,190,380,85,24);txt(s,'A relative ranking signal',900,190,330,85,24);
 txt(s,'Priority index = model risk score × annual contract value (ACV)',56,305,1168,70,30,C.orange,true);
 table(s,[['Illustrative contract','Model score','Annual value','Priority index'],['A: higher modeled risk','0.40','$50,000','20,000'],['B: higher financial exposure','0.20','$200,000','40,000']],56,405,1168,155,[440,220,250,258],23);
 foot(s,'Example numbers are illustrative. Scores are uncalibrated, so the index is a ranking tool rather than expected dollars lost.');
}
// 15. A clear review population without reseller concepts.
{
 const s=slide('Renewals for the April customer review',
 'At the April first review, I gather subscriptions expiring April second through June thirtieth. All purchasing accounts are treated as directly managed in this scenario. Existing contract, history and decision-known checks remain. Model scores were computed at each contract’s own expiry-minus-90 date, so stored scores can be zero to 89 days old at review. This is a retrospective demonstration, not a refreshed April model or a live outreach queue. The outcome becomes observable after each expiry plus 90 days. The entire latest cohort has completed follow-up by September 28. Manual contact checks still cover prior contact history, measurement quality, active escalation and urgency. The same review pool is used by every policy. Target 1.5 minutes.',
 'part2/outputs/cohort_eligibility_waterfall.csv; timing_audit.json; customer_management_scope.md','Risk to action');
 label(s,'Review date',56,145,320);txt(s,'1 April 2026',56,205,420,80,39,C.ink,true);
 label(s,'Expiry window',675,145,480);txt(s,'2 April to 30 June 2026',675,205,535,85,35,C.ink,true);
 txt(s,fmt(b.portfolio.n_review_eligible)+' subscription contracts',56,345,1168,70,39,C.orange,true);
 txt(s,'Across '+fmt(b.portfolio.n_review_eligible_accounts)+' purchasing accounts',56,425,1168,60,30,C.ink,true);
 txt(s,'Before contact: confirm recent outreach, measurement quality, open escalations and renewal urgency.',56,535,1168,85,26);
 foot(s,'All accounts are assumed directly managed. Stored scores can be up to 89 days old when the April review begins.');
}
// Same review capacity and population across every policy.
{
 const s=slide('The model finds more cancellations in 30 reviews',
 'All policies receive the same eligible pool and exactly 30 account reviews. Each policy selects its highest-ranked contract per account. Precision is selected lead contracts later cancelled divided by 30. The model identifies '+b.policies.model_score.n_cancelled+' cancelled leads, versus '+b.policies.health_rules.n_cancelled+' for the fixed health rules. Score times ACV identifies '+b.policies.model_x_acv.n_cancelled+' cancelled leads but captures '+b.policies.model_x_acv.acv_pct+' percent of cancelled contract value. That is a financial-priority trade-off, not the best precision. Captured cancelled value is observed afterward and is not revenue saved. The paired parent-bootstrap 90 percent interval for model-minus-rule precision is '+extra.model_minus_rule_90ci_pp.join(' to ')+' percentage points. This describes uncertainty in this synthetic population, not real-world validation.',
 'part2/outputs/renewal_review_policy_comparison.csv; renewal_review_policy_diffs.csv','Risk to action');
 const keys=['health_rules','model_score','model_x_acv','commercial_value'],names=['Customer-health rules','Model risk score','Model score × ACV','Annual contract value only'];
 table(s,[['Ranking method','Cancelled leads','Precision','Cancelled value captured'],...keys.map((k,i)=>[names[i],b.policies[k].n_cancelled+' / 30',b.policies[k].precision.toFixed(1)+'%',b.policies[k].acv_pct.toFixed(1)+'%'])],56,150,1168,310,[370,230,180,388],24);
 txt(s,'Model score finds more cancellations. Score × ACV prioritizes financial exposure.',56,500,1168,80,28,C.teal,true);
 txt(s,'Model minus rules: +'+(b.policies.model_score.precision-b.policies.health_rules.precision).toFixed(1)+' percentage points.\nPaired 90% interval: +'+extra.model_minus_rule_90ci_pp[0].toFixed(1)+' to +'+extra.model_minus_rule_90ci_pp[1].toFixed(1)+' points.',56,590,1168,60,23,C.muted);
 foot(s,'Historical outcomes on synthetic histories. Cancelled value captured does not mean revenue saved.');
}
// Frozen primary model, without refitting on replication populations.
{
 const s=slide('The advantage repeats in fresh synthetic populations',
 'The same primary fitted model and preprocessing score every population. Seed 137 is the primary dataset. Seeds 991 and 2027 generate separate 10,000-account populations under the same frozen mechanisms. No model refit or seed selection occurs on the replications. At 30 reviews, model versus rules finds 13 versus 3 cancelled leads in the primary population, 16 versus 8 in replication one, and 9 versus 7 in replication two. All predeclared populations are shown. The smaller advantage in the second replication demonstrates variation. All three use the same chosen mechanisms, so this tests recovery of the simulation rather than transfer to real Palo Alto Networks customers. It does not establish that the model wins in every future cohort.',
 'output/model_demo_revision/all_seed_policy_results.csv; all_seed_test_metrics.csv; replication_receipt.json','Model validation');
 const seeds=[137,991,2027];
 const get=(seed,policy)=>b.replications.find(r=>r.seed===seed&&r.policy===policy);
 chart(s,'bar',{position:{left:40,top:140,width:790,height:445},title:'Precision at 30 account reviews',categories:['Primary','Replication 1','Replication 2'],series:[
 {name:'Model score',values:seeds.map(v=>get(v,'model_score').precision_lead_churned),fill:C.teal},
 {name:'Health rules',values:seeds.map(v=>get(v,'health_rules').precision_lead_churned),fill:C.orange}],
 barOptions:{direction:'column',grouping:'clustered',gapWidth:90},legend:{position:'bottom',textStyle:{fontSize:18}},xAxis:{title:'Separately generated population',textStyle:{fontSize:18}},yAxis:{min:0,max:.6,numberFormatCode:'0%',title:'Cancelled leads per 30 reviews',textStyle:{fontSize:18}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0%',textStyle:{fontSize:18}},chartFill:C.white});
 label(s,'One frozen model',870,155,340);
 txt(s,'Primary: 13 vs 3\nReplication 1: 16 vs 8\nReplication 2: 9 vs 7',870,230,335,190,29,C.teal,true);
 txt(s,'No refitting on either replication.\nAll three populations reported.',870,465,335,105,24);
 foot(s,'Seeds 137, 991 and 2027. Each population has 10,000 generated accounts. Shared simulation assumptions limit real-world conclusions.');
}

// Top model-ranked lead, selected without using its outcome.
{
 const e=b.example_account;
 const date=new Date(e.term_end+'T00:00:00Z').toLocaleDateString('en-GB',{day:'numeric',month:'long',year:'numeric',timeZone:'UTC'});
 const s=slide('A high-risk account and a practical response',
 'This is the first-ranked lead under model score, selected without filtering on its outcome. Account '+e.purchasing_account+' holds '+e.product_name+', with expiry '+date+'. Only '+e.deployment_reach_30d+' percent of intended scope is deployed and '+e.workflow_success_30d+' percent of workflow attempts succeed. There are '+e.p1_tickets_90d+' recent critical tickets and the recorded budget concern is '+e.dated_budget_concern+'. Do not invent a service or budget problem when these are absent. The combined adoption and successful-use evidence motivates a deployment-recovery conversation. Confirm that the measurements still describe the account, identify implementation barriers, agree a named owner and a measurable pre-expiry milestone. At the April review the stored score is '+e.score_age_days+' days old, with '+e.days_to_expiry_at_review+' days until expiry. The retrospective outcome is '+(e.churned?'recorded cancellation':'no recorded cancellation')+'. No intervention or saved revenue is simulated.',
 'part2/outputs/renewal_review_contracts.csv; renewal_review_accounts.csv; account '+e.purchasing_account,'Risk to action');
 body(s,'Account '+e.purchasing_account+'   '+e.product_name+'\nRenewal: '+date,120,90,27);
 table(s,[['What we observe','What the customer-success manager should ask'],[e.deployment_reach_30d+'% of intended scope deployed','What prevents deployment across the agreed scope?'],[e.workflow_success_30d+'% of workflow attempts succeed','Which workflows fail and what blocks successful use?'],[e.p1_tickets_90d+' critical tickets and '+(e.dated_budget_concern?'budget concern recorded':'no recorded budget concern'),'Confirm the implementation issue before proposing an action.']],56,245,1168,260,[440,728],24);
 txt(s,'Proposed action: agree a deployment milestone, success target and owner before expiry.',56,550,1168,65,28,C.teal,true);
 foot(s,'Score '+e.model_score.toFixed(3)+'. Annual value $'+fmt(e.acv)+'. Retrospective outcome: '+(e.churned?'recorded cancellation':'no recorded cancellation')+'. An account example does not establish treatment impact.');
}

// 18. Financial context adds sales-growth lever.
{
 const s=slide('Financial retention and expansion',
 'The model ranks subscription cancellation risk. Financial retention separately measures annualized value from opening customers. Opening annual recurring value is '+b.arr.opening_arr_m+' million and closing value '+b.arr.closing_arr_m+' million. Gross retention is '+b.arr.grr+' percent and net retention '+b.arr.nrr+' percent. Closing total can grow while opening customers contract because new business adds '+b.arr.new_business_m+' million. This bridge covers October 2024 through September 2025 and is not PANW fiscal reporting or recognized revenue. Full-history expansion is '+b.growth.full_hist_expansion_m+' million, contraction '+b.growth.full_hist_contraction_m+' million. Quantity-dominant event categories account for '+b.growth.qty_dominant_pct+' percent of expansion. These descriptive patterns motivate capacity reviews among healthy adopters. They do not come from the churn model and do not prove a causal growth lever. All amounts are synthetic.',
 'part2/outputs/arr_bridge.csv; analyze_growth_summary.json','Business impact');
 table(s,[['Financial lens','Observed synthetic result','Business implication'],['Gross revenue retention (GRR)',b.arr.grr.toFixed(2)+'%','Opening value falls through losses and reductions'],['Net revenue retention (NRR)',b.arr.nrr.toFixed(2)+'%','Opening customers finish below their starting annual value'],['Full-history expansion','$'+b.growth.full_hist_expansion_m+'M','A separate opportunity alongside preventing losses'],['Quantity-dominant expansion',b.growth.qty_dominant_pct.toFixed(1)+'% of expansion value','Review adoption and capacity before offering more']],56,150,1168,330,[370,260,538],23);
 txt(s,'Annual recurring value: $'+b.arr.opening_arr_m+'M opening and $'+b.arr.closing_arr_m+'M closing',56,530,1168,75,28,C.teal,true);
 foot(s,'The snapshot bridge covers October 2024 to September 2025. Expansion uses the full history. Synthetic amounts are not PANW financial estimates.');
}
// 19. Recommendations explicitly linked to evidence and measurement.
{
 const best=Object.entries(b.policies).sort((a,z)=>z[1].n_cancelled-a[1].n_cancelled)[0][0];
 const names={health_rules:'Health rules',model_score:'Model score',model_x_acv:'Score × ACV',commercial_value:'Annual value'};
 const s=slide('Recommended actions and success measures',
 'These are proposed actions supported by patterns in synthetic evidence, not proven retention levers. First, examine delayed initial value, poor protection and failed workflows before asking the customer to buy more. Agree milestones and measure recovery. Second, benchmark any model ranking against simple health rules at the actual capacity. Under the current results, '+names[best]+' has the highest observed cancellation count at 30 reviews; the frozen model is also checked on two fresh synthetic populations. Real-data validation is still needed. Third, review adoption and unused capacity before expansion conversations because quantity-dominant events account for most historical expansion value. A qualified customer should experience a useful action, not merely receive a high risk label. Measure incremental value and cancellation outcomes in a controlled pilot. Target 2 minutes.',
 'feature_importance.csv; renewal_review_policy_comparison.csv; analyze_growth_summary.json; transcript 00:31 to 01:43 and 10:51 to 12:21','Recommendations');
 table(s,[['Recommendation','Evidence in this study','How to measure success'],['Product-recovery review','Workflow success and protection coverage lead model importance','Agreed milestones reached and service issues resolved'],['A capacity-based outreach policy','Different rankings select different 30-account lists','Cancelled leads found, staff time and outcome uncertainty'],['An adoption and capacity review before expansion',b.growth.qty_dominant_pct+'% of expansion sits in quantity-dominant categories','Verified additional annual value and sustained product adoption']],56,150,1168,350,[360,360,448],24);
 body(s,'The next step is a measurable customer action, followed by a fair comparison.',545,70,28);
 foot(s,'Proposals are hypotheses. A model score alone cannot establish an intervention effect or revenue saved.');
}
// 20. End with actual decision and next steps rather than another summary slide.
{
 const s=slide('Validation, pilot and next steps',
 'Separate prediction quality from intervention effectiveness. The frozen model performs better than health rules on all three reported synthetic populations. Next validate the assumptions and ranking on real untreated customer data before deployment. Next test whether the customer-success team can deliver the agreed recovery actions. Screen every candidate before assignment, coordinate or randomize by parent organization, freeze the lead contract and record usual care. Follow cancellation through expiry plus 90 days and distinguish verified renewal value from no cancellation event. A small pilot mainly tests feasibility. A powered effect claim requires an enrolled baseline rate, plausible effect size and allowance for parent clustering. The appendix gives an illustrative sample-size calculation. Close by asking about actual review capacity, preferred business objective and feasible interventions. Target 2 minutes plus discussion.',
 'part2/docs/pilot_protocol.md; transcript 10:51 to 12:21 and 13:23 to 13:45','Next steps');
 label(s,'Prediction quality',56,145,500);txt(s,'Freeze the model and review policy.\nEvaluate on fresh, untreated observations.\nConfirm performance on real customer data.',56,220,535,190,27);
 label(s,'Customer outcomes',700,145,500);txt(s,'Check readiness before contacting accounts.\nPilot a documented recovery action.\nFollow cancellation through expiry + 90 days.',700,220,520,190,27);
 txt(s,'Discussion: which objective, review capacity and interventions should guide the pilot?',56,505,1168,100,31,C.orange,true);
 foot(s,'Synthetic results demonstrate ranking behavior. A future controlled pilot must test whether actions improve actual retention.');
}
// Close the main narrative before opening supporting material.
{
 const s=slide('Discussion',
 'Thank the audience and invite questions. The main presentation ends here. Use the appendix for deeper questions about simulation assumptions, model evaluation, financial definitions and pilot design.',
 'Closing discussion','Closing');
 s.background.fill=C.navy;
 txt(s,'Thank you',70,220,1100,100,66,C.white,true);
 txt(s,'Questions and discussion',70,355,1100,65,34,C.orange);
}
// Appendix: simulation outcome check, moved from the main narrative.
{
 const s=slide('Appendix: simulation outcome check',
 'This is a simulation validation check, not an independently discovered driver. Higher baseline risk for smaller accounts is a declared generation assumption. This chart uses actual mature contract observations retained in temporal splits. It differs from the annual rate assumption chart because it mixes term lengths, customer experiences and years. There are '+fmt(extra.retained_labeled)+' labeled observations and '+fmt(extra.positives)+' recorded cancellations. An always-no-cancellation classifier would look accurate but miss every cancellation. That is why I report ranking quality and results at a fixed review capacity. Size is useful context under the declared simulator assumptions. It is not a reason to contact a customer without checking their product experience. The large-enterprise group has very few cancellations, so its rate is especially uncertain. Optional discussion only.',
 'part2/outputs/split_cohort_full.csv; extra_evidence.json','Appendix');
 const order=['smb','midmarket','enterprise','large_enterprise'];const names=['Small and medium business','Midmarket','Enterprise','Large enterprise'];
 const rr=order.map(k=>extra.segments.find(x=>x.size_segment===k));
 bars(s,'Observed cancellations per contract',names,rr.map(x=>x.rate),45,145,760,410);
 label(s,fmt(extra.positives)+' cancellations',845,160,370,C.orange);
 txt(s,'Across '+fmt(extra.retained_labeled)+' mature observations\n\nAbout '+(100*extra.positives/extra.retained_labeled).toFixed(1)+'% of contracts\n\nA model must do more than predict the common outcome.',845,225,365,290,25);
 foot(s,'Observed contract rates differ from annual anchors. The large-enterprise group has '+fmt(extra.segments.find(r=>r.size_segment==='large_enterprise').positives)+' recorded cancellation.');
}
// Appendix 1, full mandatory EDA table.
{
 const s=slide('Appendix: top-20 products by order count',
 'Ranks use distinct eligible orders containing each product. Companies means distinct purchasing accounts in the source output. Support, hardware and services remain in purchasing EDA even though the churn model focuses on security subscriptions. Units differ by product, so these counts should not be compared with license quantities.',
 'outputs/eda_top20_ordered_items.csv','Appendix');
 for(let half=0;half<2;half++)table(s,[['No.','Product','Orders'],...b.eda.top20_products.slice(half*10,half*10+10).map(r=>[r.rank,r.product,fmt(r.orders)])],56+half*594,133,574,500,[70,384,120],20);
 foot(s,'Distinct eligible order count. Support products remain part of purchasing EDA, while the cancellation model uses security subscriptions.');
}
// Appendix 3, actual baseline formula as distinct from the ML model.
{
 const s=slide('Appendix: health rules and model scoring',
 'The baseline is a manually weighted health score. It is separate from the selected trained model. Deployment gap equals one minus deployment reach, clipped between zero and one. Poor workflow results equal one minus workflow success. Service pain combines normalized unresolved impact hours at 60 percent and breached tickets at 40 percent. The hour normalization caps at 720 and the breach normalization at three. Budget concern is the recorded binary flag. The four rule weights are fixed and fixed across every reported population. The generator includes additional declared interactions, while this additive comparator remains a transparent benchmark. The current implementation fills missing deployment reach with zero before subtracting from one, which yields maximum gap, while missing workflow success gets neutral 0.5. Do not claim missing deployment reach gets zero penalty. The learned HGB uses data-fitted tree rules, not these four weights. Its output is a logistic transformation of its additive tree score. Target only if asked.',
 'part2/src/_rule_baseline.py; model_selection_record.json','Appendix');
 txt(s,'Health score =',56,145,1168,50,31,C.orange,true);
 table(s,[['Component','Weight','Calculation'],['Deployment gap','35%','1 minus deployment reach'],['Poor workflow results','30%','1 minus workflow success'],['Service pain','25%','60% impact-hours score + 40% breached-ticket score'],['Recorded budget concern','10%','1 if a dated concern exists, otherwise 0']],56,220,1168,290,[330,140,698],24);
 txt(s,'Learned model: 100 shallow trees, maximum depth 3, learning rate 0.05',56,550,1168,65,25,C.teal,true);
 s.speakerNotes.text+=' For boosted trees, F(x) = b + 0.05 × sum of 100 learned tree outputs, with score = 1 / (1 + exp(−F(x))). The intercept and tree rules are fitted from historical labels. The separately defined health score uses the manual weights above.';
 foot(s,'Scores support relative ranking. Missing deployment reach currently receives a full gap penalty. Both scoring approaches remain uncalibrated.');
}
// Appendix 4, robustness and honest uncertainty.
{
 const s=slide('Appendix: robustness and uncertainty',
 'Ablation means retraining with one group of inputs removed while holding classifier settings and population fixed. A positive change here means the simpler model improved AP in this development period. Correlated inputs can substitute for one another. None of these tests prove causal business importance. The unseen-parent sensitivity excludes parents already represented in both training and selection, which is the appropriate comparison for the refitted artifact. A more favorable sensitivity AP is not evidence of performance on real customers. Paired bootstrap comparisons resample parent groups and preserve repeated draws, with the same draw across policies. The primary model-versus-rule precision interval is positive in this synthetic review. This does not quantify uncertainty about real customer behavior.',
 'part2/outputs/ablation.csv; metrics.json; renewal_review_policy_diffs.csv','Appendix');
 const names={block_deployment:'Deployment',block_value:'Productive use',block_telemetry:'Coverage',block_service:'Support',block_commercial:'Commercial',block_relationship:'Relationship'};
 table(s,[['Feature block removed','AP without block','AP change'],...b.ablation.map(r=>[names[r.block],ap(r.ablated_ap),((-Number(r.ap_drop)>=0)?'+':'')+(-Number(r.ap_drop)).toFixed(4)])],56,145,1168,330,[500,280,388],23);
 txt(s,'Unseen-parent check: '+fmt(b.unseen.n_clean_fit)+' observations, '+b.unseen.pos_clean_fit+' cancellations. Refit AP '+ap(b.unseen.hgb_refitted_ap_unseen_fit)+'.',56,525,1168,80,25);
 foot(s,'Same model settings for block tests. A sensitivity result or a favorable point estimate does not establish a robust business improvement.');
}
// Appendix 5, all signed financial components retained.
{
 const a=b.arr;
 const s=slide('Subscription value reconciliation',
 'This reconciles opening annualized subscription value to closing value for 1 October 2024 through 30 September 2025. Account loss means no active closing subscriptions, which is a snapshot definition and need not equal confirmed cancellation events. Partial loss means absent opening product value while the account stays active. Contraction is a separate lower-value renewal effect. Renewal gaps and unresolved lineage remain explicit rather than relabeled as verified new purchases. The bridge reconciles to zero using unrounded source values. Displayed millions may differ by rounding. This reporting window is not PANW’s fiscal year. GRR and NRR describe financial retention, distinct from the cancellation target.',
 'part2/outputs/arr_bridge.csv; lineage_reconciliation.csv','Appendix');
 table(s,[['Component','Change / value ($M)'],['Opening annual recurring value',a.opening_arr_m.toFixed(1)],['Account loss','−'+a.account_churn_m.toFixed(1)],['Partial product loss','−'+a.partial_loss_m.toFixed(1)],['Expansion','+'+a.expansion_m.toFixed(1)],['Contraction',a.contraction_m.toFixed(1)],['New business','+'+a.new_business_m.toFixed(1)],['Reactivation','+'+a.reactivation_m.toFixed(1)],['Renewal gaps','+'+a.renewal_gap_m.toFixed(2)],['Unresolved lineage','+'+a.intra_unresolved_m.toFixed(2)],['Residual','0.00'],['Closing annual recurring value',a.closing_arr_m.toFixed(1)]],56,125,1168,490,[800,368],21);
 foot(s,'October 2024 to September 2025. Snapshot losses differ from confirmed cancellation events. Unrounded source bridge reconciles to zero.');
}
// Appendix 6, practical controlled intervention and power.
{
 const s=slide('Appendix: pilot design and sample size',
 'Plan a prospective pilot with a fresh cohort. Screen all candidates before treatment assignment and freeze one lead contract per account. Randomize by parent if related accounts could share customer-success work. Compare the agreed action with usual care and analyze assignment as allocated. For urgent renewals within 30 days, contact by day three and agree an action before expiry minus seven days. The standard track contacts by day 14 and completes the agreed action by day 21 to 28. Measure delivery first. Under illustrative independent-unit assumptions, 15 percent cancellation reduced to 9 percent requires approximately 460 accounts per arm at two-sided 5 percent significance and 80 percent power. Actual enrollment, effect size, clustering and attrition determine feasibility; a smaller pilot cannot claim that level of statistical power. No treatment effect is simulated.',
 'part2/docs/pilot_protocol.md; presentation_data_bundle.json pilot_power','Appendix');
 table(s,[['Design choice','Working proposal'],['Assignment','Screen first, then randomize by parent organization'],['Treatment','A named customer-success owner and an agreed recovery milestone'],['Primary outcome','Recorded cancellation through expiry + 90 days'],['Delivery measures','Timely contact, action completed, staff time and product recovery'],['Illustrative power','15% to 9% cancellation requires about '+b.pilot_power.n_per_arm_independent+' independent accounts per arm'],['Interpretation','A small pilot tests delivery. A larger trial is needed to estimate effect.']],56,145,1168,390,[330,838],23);
 foot(s,'The 15% baseline and 6 percentage-point improvement are assumptions. Recalculate for enrollment, parent clustering and attrition.');
}
// Appendix 7, terms used in the conversation.
{
 const s=slide('Appendix: business and modeling terms',
 'These definitions are intended for a mixed technical and business audience. P1 here refers to a critical support ticket. Account review capacity is specified as 30 reviews, not as a P1 label. Precision and recall answer different questions. F1 combines both at a chosen threshold but does not encode review capacity or contract value. AP summarizes ranking across many cutoffs. A customer-success manager owns a proposed action, subject to real operating responsibilities. A lead is a contract chosen to represent one account under a specific ranking, not a sales prospect.',
 'Case-study metric and feature contracts','Appendix');
 table(s,[['Term','Plain meaning'],['ACV: annual contract value','Subscription contract value expressed per year'],['ARR: annual recurring revenue','Annual run rate of active subscriptions at a reporting snapshot'],['GRR / NRR','Opening-customer annual value retained without / with expansion'],['Precision / recall','Cancellations per review. Share of cancellations reached.'],['Average precision (AP)','A summary of precision as the ranking recovers more cancellations'],['P1 ticket / CSM','A critical support case / customer-success manager'],['Lead contract','The highest-ranked contract used to represent an account for review']],56,140,1168,445,[380,788],23);
 foot(s,'An uncalibrated risk score orders contracts. It does not establish the probability of cancellation or the amount an intervention could save.');
}

// Main narrative: approximately 35 minutes, with discussion to fill 45 minutes.
const timing=[40,90,75,100,90,110,90,90,90,60,60,60,120,110,120,120,70,150,110,100,90,90,100,15];
for(let i=0;i<p.slides.items.length;i++){
 const n=p.slides.items[i].speakerNotes;
 n.text=n.text.replace(/Target [^.!?]*[.!?]/g,'').replace(/Optional discussion only\./g,'');
 n.text+='\nSuggested delivery: '+(i<24?timing[i]+' seconds.':'Appendix for questions only.');
}
// Clean every authored string and inspect notes before exporting.
for(const s of p.slides.items){
 if(!s.speakerNotes.text?.trim())throw new Error('Missing notes');
 if(/[\u2014]/.test(s.speakerNotes.text))throw new Error('Em dash in notes');
}
await fs.mkdir(path.join(staging,'rendered'),{recursive:true});
const inspect=await p.inspect({kind:'slide,textbox,table,chart,notes',maxChars:1000000});
await fs.writeFile(path.join(staging,'deck_inspection.ndjson'),inspect.ndjson);
const candidate=path.join(staging,'story_candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(staging,'rendered',`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const hash=async f=>createHash('sha256').update(await fs.readFile(path.join(root,f))).digest('hex');
await fs.writeFile(path.join(staging,'feature_inventory.json'),JSON.stringify(featureGroups,null,2));
await fs.writeFile(path.join(staging,'storyboard.json'),JSON.stringify({slides:story,mainSlides:24,slideCount:story.length,chartOwners,tableOwners:[...new Set(tableOwners)],candidate,analysisManifestSha256:await hash('part2/outputs/run_manifest.json'),bundleSha256:await hash('part3/model_demo_presentation_data.json')},null,2));
console.log(JSON.stringify({slides:story.length,candidate,chartOwners,tableOwners:[...new Set(tableOwners)]}));
