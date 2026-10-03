subject_fixture <- function() {
  x <- ml_fixture(array(seq_len(16*2*4),c(16,2,4)),case_ids=paste0("C",1:4))
  d <- data.frame(case_id=paste0("C",1:4),subject_id=c("P1","P1","P2","P3"),
    session_id=c("V1","V2","V1","V1"),trial_id="T1",cycle_id=paste0("cycle",1:4))
  list(x=withCaseData(x,d[c(4,2,1,3),]),d=d)
}
subject_contract <- function(x, cases) {
  r <- .ml_resolve_input(x,cases=cases)
  p <- .ml_window_plan(r,8L,8L)
  .ml_plain_torch_contract(list(task="regression",channel_ids=r$channel_data$channel_id,
    window_samples=8L,dtype="float32",normalization_method="none",class_levels=NULL,
    case_ids=r$case_data$case_id,case_data=r$case_data,window_table=p$table,
    normalization_stats=NULL,normalization_fitted=FALSE))
}

test_that("subject identity is matched by case IDs and follows reordered selection", {
  f <- subject_fixture(); r <- .ml_resolve_input(f$x,cases=c("C3","C1"))
  expect_identical(r$case_data$subject_id,c("P2","P1"))
  expect_identical(r$case_data$session_id,c("V1","V1"))
  expect_identical(r$case_data$input_case_index,c(3L,1L))
  expect_equal(unname(r$data[1,,]),unname(t(SummarizedExperiment::assay(f$x)[,,3])))
  p <- .ml_window_plan(r,8L,8L)
  expect_identical(p$table$subject_id,c("P2","P2","P1","P1"))
  expect_identical(p$table$case_id,c("C3","C3","C1","C1"))
})

test_that("invalid or ambiguous case mappings are rejected", {
  f <- subject_fixture()
  expect_error(withCaseData(f$x,f$d[-1,]),"match all")
  d<-f$d;d$case_id[1]<-"C2"
  expect_error(withCaseData(f$x,d),"unique")
  d<-f$d;d$subject_id[1]<-NA_character_
  expect_error(withCaseData(f$x,d),"complete")
  d<-f$d;d$subject_id<-factor(d$subject_id)
  expect_error(withCaseData(f$x,d),"character")
  d<-f$d;d$participant_id<-"wrong"
  expect_error(withCaseData(f$x,d),"equal subject_id")
  # Invalid table also fails at use time, not just through the attachment helper.
  md<-S4Vectors::metadata(f$x);md$ml_case_data<-f$d[-1,]
  S4Vectors::metadata(f$x)<-md
  expect_error(.ml_resolve_input(f$x),"match all")
})

test_that("different cases from the same subject fail the shared training guard", {
  f<-subject_fixture()
  a<-list(contract=subject_contract(f$x,"C1"))
  b<-list(contract=subject_contract(f$x,"C2"))
  expect_error(.ml_compare_dataset_contracts(a,b),"subject IDs must be disjoint; shared: P1")
  b<-list(contract=subject_contract(f$x,c("C3","C4")))
  expect_true(.ml_compare_dataset_contracts(a,b))
  b$contract$case_data<-NULL
  expect_error(.ml_compare_dataset_contracts(a,b),"Both train and valid")
  a$contract$case_data<-NULL
  expect_true(.ml_compare_dataset_contracts(a,b))
})

test_that("serializable contracts retain subject keys and rejection behavior", {
  f<-subject_fixture(); a<-subject_contract(f$x,"C1"); b<-subject_contract(f$x,"C2")
  path<-tempfile(fileext=".rds");on.exit(unlink(path))
  saveRDS(list(a=a,b=b),path); back<-readRDS(path)
  expect_identical(back$a$case_data,a$case_data)
  expect_identical(back$a$window_table$subject_id,c("P1","P1"))
  expect_error(.ml_compare_dataset_contracts(list(contract=back$a),list(contract=back$b)),"subject IDs")
})

test_that("feature rows use the same subject split guard without becoming time samples", {
  f<-subject_fixture(); m<-matrix(1:8,4,dimnames=list(f$d$case_id,c("emg","motion")))
  v<-withCaseData(m,f$d[4:1,])
  expect_identical(attr(v,"case_data"),f$d)
  expect_equal(unname(v[,1]),1:4)
  expect_error(validateSubjectSplit(f$d[1,,drop=FALSE],f$d[2,,drop=FALSE]),"subject IDs")
  expect_true(validateSubjectSplit(f$d[1:2,],f$d[3:4,]))
  expect_error(validateSubjectSplit(f$d[1:2,],f$d[2:3,]),"case IDs")
})

test_that("catch22 and reduced matrices retain case metadata outside feature columns", {
  skip_if_not_installed("Rcatch22")
  f<-subject_fixture()
  result<-catch22(f$x,cases=c("C3","C1"))
  expect_identical(attr(result,"case_data")$subject_id,c("P2","P1"))
  mat<-peReducedFeatures(f$x,method="catch22",cases=c("C3","C1"))
  expect_identical(attr(mat,"case_data")$subject_id,c("P2","P1"))
  expect_false(any(grepl("subject_id",colnames(mat),fixed=TRUE)))
})

test_that("actual torch datasets carry subjects into training-time rejection", {
  ml_skip_torch(require_luz=TRUE)
  f<-subject_fixture()
  a<-peDataset(f$x,cases="C1",targets=1,task="regression",window_samples=8)
  b<-peDataset(f$x,cases="C2",targets=2,task="regression",window_samples=8)
  expect_identical(a$contract$window_table$subject_id,c("P1","P1"))
  expect_error(trainModel(a,b,model="cnn1d",epochs=1),"subject IDs")
})


test_that("transform-produced case tables can be validated and reattached", {
  f<-subject_fixture(); a<-.ml_resolve_input(f$x,cases=c("C2","C1"))$case_data
  b<-.ml_resolve_input(f$x,cases=c("C3","C4"))$case_data
  expect_true(validateSubjectSplit(a,b))
  m<-matrix(1:4,2,dimnames=list(a$case_id,c("emg","motion")))
  expect_identical(attr(withCaseData(m,a),"case_data"),a)
  full<-.ml_resolve_input(f$x)$case_data;full$input_case_index<-11:14
  expect_identical(.ml_resolve_input(withCaseData(f$x,full),cases="C2")$case_data$input_case_index,2L)
  full$input_case_index[1]<-NA_integer_
  expect_error(withCaseData(f$x,full),"positive whole")
})


test_that("identity columns cannot contain nested matrices", {
  f<-subject_fixture();d<-f$d
  d$subject_id<-I(matrix(rep(c("P1","P2"),each=4),nrow=4))
  expect_error(withCaseData(f$x,d),"character labels")
  d<-f$d;d$input_case_index<-I(matrix(1:8,nrow=4))
  expect_error(withCaseData(f$x,d),"positive whole")
})

test_that("tabular prediction formatting retains window subject/session keys", {
  f<-subject_fixture();contract<-subject_contract(f$x,c("C3","C1"))
  model<-list(task="regression",model_spec=list(n_outputs=1L))
  answer<-.ml_format_predictions(1:4,model,list(contract=contract),"response")
  expect_identical(answer$subject_id,c("P2","P2","P1","P1"))
  expect_identical(answer$session_id,contract$window_table$session_id)
  expect_equal(answer$response_1,1:4)
})
