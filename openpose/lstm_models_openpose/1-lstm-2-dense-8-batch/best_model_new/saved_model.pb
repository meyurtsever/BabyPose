ԫ/
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??-
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:  *
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
: *
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:  *
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
: *
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

: *
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
?
lstm_5/lstm_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	&?*+
shared_namelstm_5/lstm_cell_10/kernel
?
.lstm_5/lstm_cell_10/kernel/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/kernel*
_output_shapes
:	&?*
dtype0
?
$lstm_5/lstm_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*5
shared_name&$lstm_5/lstm_cell_10/recurrent_kernel
?
8lstm_5/lstm_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp$lstm_5/lstm_cell_10/recurrent_kernel*
_output_shapes
:	 ?*
dtype0
?
lstm_5/lstm_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelstm_5/lstm_cell_10/bias
?
,lstm_5/lstm_cell_10/bias/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
~
dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *"
shared_namedense_15/kernel/m
w
%dense_15/kernel/m/Read/ReadVariableOpReadVariableOpdense_15/kernel/m*
_output_shapes

:  *
dtype0
v
dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedense_15/bias/m
o
#dense_15/bias/m/Read/ReadVariableOpReadVariableOpdense_15/bias/m*
_output_shapes
: *
dtype0
~
dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *"
shared_namedense_16/kernel/m
w
%dense_16/kernel/m/Read/ReadVariableOpReadVariableOpdense_16/kernel/m*
_output_shapes

:  *
dtype0
v
dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedense_16/bias/m
o
#dense_16/bias/m/Read/ReadVariableOpReadVariableOpdense_16/bias/m*
_output_shapes
: *
dtype0
~
dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namedense_17/kernel/m
w
%dense_17/kernel/m/Read/ReadVariableOpReadVariableOpdense_17/kernel/m*
_output_shapes

: *
dtype0
v
dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_17/bias/m
o
#dense_17/bias/m/Read/ReadVariableOpReadVariableOpdense_17/bias/m*
_output_shapes
:*
dtype0
?
lstm_5/lstm_cell_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	&?*-
shared_namelstm_5/lstm_cell_10/kernel/m
?
0lstm_5/lstm_cell_10/kernel/m/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/kernel/m*
_output_shapes
:	&?*
dtype0
?
&lstm_5/lstm_cell_10/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*7
shared_name(&lstm_5/lstm_cell_10/recurrent_kernel/m
?
:lstm_5/lstm_cell_10/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&lstm_5/lstm_cell_10/recurrent_kernel/m*
_output_shapes
:	 ?*
dtype0
?
lstm_5/lstm_cell_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelstm_5/lstm_cell_10/bias/m
?
.lstm_5/lstm_cell_10/bias/m/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/bias/m*
_output_shapes	
:?*
dtype0
~
dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *"
shared_namedense_15/kernel/v
w
%dense_15/kernel/v/Read/ReadVariableOpReadVariableOpdense_15/kernel/v*
_output_shapes

:  *
dtype0
v
dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedense_15/bias/v
o
#dense_15/bias/v/Read/ReadVariableOpReadVariableOpdense_15/bias/v*
_output_shapes
: *
dtype0
~
dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *"
shared_namedense_16/kernel/v
w
%dense_16/kernel/v/Read/ReadVariableOpReadVariableOpdense_16/kernel/v*
_output_shapes

:  *
dtype0
v
dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedense_16/bias/v
o
#dense_16/bias/v/Read/ReadVariableOpReadVariableOpdense_16/bias/v*
_output_shapes
: *
dtype0
~
dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namedense_17/kernel/v
w
%dense_17/kernel/v/Read/ReadVariableOpReadVariableOpdense_17/kernel/v*
_output_shapes

: *
dtype0
v
dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_17/bias/v
o
#dense_17/bias/v/Read/ReadVariableOpReadVariableOpdense_17/bias/v*
_output_shapes
:*
dtype0
?
lstm_5/lstm_cell_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	&?*-
shared_namelstm_5/lstm_cell_10/kernel/v
?
0lstm_5/lstm_cell_10/kernel/v/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/kernel/v*
_output_shapes
:	&?*
dtype0
?
&lstm_5/lstm_cell_10/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*7
shared_name(&lstm_5/lstm_cell_10/recurrent_kernel/v
?
:lstm_5/lstm_cell_10/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&lstm_5/lstm_cell_10/recurrent_kernel/v*
_output_shapes
:	 ?*
dtype0
?
lstm_5/lstm_cell_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelstm_5/lstm_cell_10/bias/v
?
.lstm_5/lstm_cell_10/bias/v/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_10/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?6
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?5
value?5B?5 B?5
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
?
(iter

)beta_1

*beta_2
	+decay
,learning_ratemcmdmemf"mg#mh-mi.mj/mkvlvmvnvo"vp#vq-vr.vs/vt
?
-0
.1
/2
3
4
5
6
"7
#8
?
-0
.1
/2
3
4
5
6
"7
#8
 
?
trainable_variables
	variables
0layer_regularization_losses
1non_trainable_variables
2metrics
	regularization_losses
3layer_metrics

4layers
 
~

-kernel
.recurrent_kernel
/bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
 

-0
.1
/2

-0
.1
/2
 
?
trainable_variables
	variables
9layer_regularization_losses
:non_trainable_variables
;metrics
regularization_losses

<states
=layer_metrics

>layers
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
	variables
?layer_regularization_losses
@non_trainable_variables
Ametrics
regularization_losses
Blayer_metrics

Clayers
 
 
 
?
trainable_variables
	variables
Dlayer_regularization_losses
Enon_trainable_variables
Fmetrics
regularization_losses
Glayer_metrics

Hlayers
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
	variables
Ilayer_regularization_losses
Jnon_trainable_variables
Kmetrics
 regularization_losses
Llayer_metrics

Mlayers
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
?
$trainable_variables
%	variables
Nlayer_regularization_losses
Onon_trainable_variables
Pmetrics
&regularization_losses
Qlayer_metrics

Rlayers
CA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUElstm_5/lstm_cell_10/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE$lstm_5/lstm_cell_10/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUElstm_5/lstm_cell_10/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 

S0
T1
 
#
0
1
2
3
4

-0
.1
/2

-0
.1
/2
 
?
5trainable_variables
6	variables
Ulayer_regularization_losses
Vnon_trainable_variables
Wmetrics
7regularization_losses
Xlayer_metrics

Ylayers
 
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ztotal
	[count
\	variables
]	keras_api
D
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1

\	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1

a	variables
yw
VARIABLE_VALUEdense_15/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_15/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_16/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_16/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_17/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_17/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUElstm_5/lstm_cell_10/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&lstm_5/lstm_cell_10/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUElstm_5/lstm_cell_10/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_15/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_15/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_16/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_16/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEdense_17/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEdense_17/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUElstm_5/lstm_cell_10/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&lstm_5/lstm_cell_10/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUElstm_5/lstm_cell_10/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_5_inputPlaceholder*+
_output_shapes
:?????????&*
dtype0* 
shape:?????????&
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_5_inputlstm_5/lstm_cell_10/kernellstm_5/lstm_cell_10/bias$lstm_5/lstm_cell_10/recurrent_kerneldense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_327477
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpiter/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp.lstm_5/lstm_cell_10/kernel/Read/ReadVariableOp8lstm_5/lstm_cell_10/recurrent_kernel/Read/ReadVariableOp,lstm_5/lstm_cell_10/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp%dense_15/kernel/m/Read/ReadVariableOp#dense_15/bias/m/Read/ReadVariableOp%dense_16/kernel/m/Read/ReadVariableOp#dense_16/bias/m/Read/ReadVariableOp%dense_17/kernel/m/Read/ReadVariableOp#dense_17/bias/m/Read/ReadVariableOp0lstm_5/lstm_cell_10/kernel/m/Read/ReadVariableOp:lstm_5/lstm_cell_10/recurrent_kernel/m/Read/ReadVariableOp.lstm_5/lstm_cell_10/bias/m/Read/ReadVariableOp%dense_15/kernel/v/Read/ReadVariableOp#dense_15/bias/v/Read/ReadVariableOp%dense_16/kernel/v/Read/ReadVariableOp#dense_16/bias/v/Read/ReadVariableOp%dense_17/kernel/v/Read/ReadVariableOp#dense_17/bias/v/Read/ReadVariableOp0lstm_5/lstm_cell_10/kernel/v/Read/ReadVariableOp:lstm_5/lstm_cell_10/recurrent_kernel/v/Read/ReadVariableOp.lstm_5/lstm_cell_10/bias/v/Read/ReadVariableOpConst*1
Tin*
(2&	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_330158
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasiterbeta_1beta_2decaylearning_ratelstm_5/lstm_cell_10/kernel$lstm_5/lstm_cell_10/recurrent_kernellstm_5/lstm_cell_10/biastotalcounttotal_1count_1dense_15/kernel/mdense_15/bias/mdense_16/kernel/mdense_16/bias/mdense_17/kernel/mdense_17/bias/mlstm_5/lstm_cell_10/kernel/m&lstm_5/lstm_cell_10/recurrent_kernel/mlstm_5/lstm_cell_10/bias/mdense_15/kernel/vdense_15/bias/vdense_16/kernel/vdense_16/bias/vdense_17/kernel/vdense_17/bias/vlstm_5/lstm_cell_10/kernel/v&lstm_5/lstm_cell_10/recurrent_kernel/vlstm_5/lstm_cell_10/bias/v*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_330276݇,
?
?
%sequential_5_lstm_5_while_cond_325442D
@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counterJ
Fsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations)
%sequential_5_lstm_5_while_placeholder+
'sequential_5_lstm_5_while_placeholder_1+
'sequential_5_lstm_5_while_placeholder_2+
'sequential_5_lstm_5_while_placeholder_3F
Bsequential_5_lstm_5_while_less_sequential_5_lstm_5_strided_slice_1\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_325442___redundant_placeholder0\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_325442___redundant_placeholder1\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_325442___redundant_placeholder2\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_325442___redundant_placeholder3&
"sequential_5_lstm_5_while_identity
?
sequential_5/lstm_5/while/LessLess%sequential_5_lstm_5_while_placeholderBsequential_5_lstm_5_while_less_sequential_5_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_5/lstm_5/while/Less?
"sequential_5/lstm_5/while/IdentityIdentity"sequential_5/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_5/lstm_5/while/Identity"Q
"sequential_5_lstm_5_while_identity+sequential_5/lstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
ɯ
?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_329875

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????&2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????&2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????&2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????&2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????&2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2?ͮ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????&2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????&2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2?2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????&2
dropout_3/Mul_1^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/Const?
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_5/Const?
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_6/Const?
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shape?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_6/GreaterEqual/y?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_6/GreaterEqual?
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_6/Cast?
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_7/Const?
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shape?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_7/GreaterEqual/y?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_7/GreaterEqual?
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_7/Cast?
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:????????? 2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:????????? 2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:????????? 2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:????????? 2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:????????? 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:????????? 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:????????? 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:????????? 2
	BiasAdd_3f
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
mul_4f
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
mul_5f
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
mul_6f
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:????????? 2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:????????? 2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:????????? 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:????????? 2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
mul_10?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????&:????????? :????????? :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?[
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_326296

inputs
lstm_cell_10_326202
lstm_cell_10_326204
lstm_cell_10_326206
identity??:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?$lstm_cell_10/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_326202lstm_cell_10_326204lstm_cell_10_326206*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_3258012&
$lstm_cell_10/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_326202lstm_cell_10_326204lstm_cell_10_326206*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_326215*
condR
while_cond_326214*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_10_326202*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_10_326204*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_10/StatefulPartitionedCall^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????&
 
_user_specified_nameinputs
?
~
)__inference_dense_16_layer_call_fn_329683

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_3272122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_330276
file_prefix$
 assignvariableop_dense_15_kernel$
 assignvariableop_1_dense_15_bias&
"assignvariableop_2_dense_16_kernel$
 assignvariableop_3_dense_16_bias&
"assignvariableop_4_dense_17_kernel$
 assignvariableop_5_dense_17_bias
assignvariableop_6_iter
assignvariableop_7_beta_1
assignvariableop_8_beta_2
assignvariableop_9_decay%
!assignvariableop_10_learning_rate2
.assignvariableop_11_lstm_5_lstm_cell_10_kernel<
8assignvariableop_12_lstm_5_lstm_cell_10_recurrent_kernel0
,assignvariableop_13_lstm_5_lstm_cell_10_bias
assignvariableop_14_total
assignvariableop_15_count
assignvariableop_16_total_1
assignvariableop_17_count_1)
%assignvariableop_18_dense_15_kernel_m'
#assignvariableop_19_dense_15_bias_m)
%assignvariableop_20_dense_16_kernel_m'
#assignvariableop_21_dense_16_bias_m)
%assignvariableop_22_dense_17_kernel_m'
#assignvariableop_23_dense_17_bias_m4
0assignvariableop_24_lstm_5_lstm_cell_10_kernel_m>
:assignvariableop_25_lstm_5_lstm_cell_10_recurrent_kernel_m2
.assignvariableop_26_lstm_5_lstm_cell_10_bias_m)
%assignvariableop_27_dense_15_kernel_v'
#assignvariableop_28_dense_15_bias_v)
%assignvariableop_29_dense_16_kernel_v'
#assignvariableop_30_dense_16_bias_v)
%assignvariableop_31_dense_17_kernel_v'
#assignvariableop_32_dense_17_bias_v4
0assignvariableop_33_lstm_5_lstm_cell_10_kernel_v>
:assignvariableop_34_lstm_5_lstm_cell_10_recurrent_kernel_v2
.assignvariableop_35_lstm_5_lstm_cell_10_bias_v
identity_37??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*?
value?B?%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_16_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_16_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_17_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_17_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp.assignvariableop_11_lstm_5_lstm_cell_10_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp8assignvariableop_12_lstm_5_lstm_cell_10_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp,assignvariableop_13_lstm_5_lstm_cell_10_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_dense_15_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_15_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_dense_16_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_16_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_dense_17_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp#assignvariableop_23_dense_17_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_lstm_5_lstm_cell_10_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp:assignvariableop_25_lstm_5_lstm_cell_10_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_lstm_5_lstm_cell_10_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_dense_15_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_15_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp%assignvariableop_29_dense_16_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_16_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp%assignvariableop_31_dense_17_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_17_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_lstm_5_lstm_cell_10_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp:assignvariableop_34_lstm_5_lstm_cell_10_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp.assignvariableop_35_lstm_5_lstm_cell_10_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_359
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_36?
Identity_37IdentityIdentity_36:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_37"#
identity_37Identity_37:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
while_cond_329445
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_329445___redundant_placeholder04
0while_while_cond_329445___redundant_placeholder14
0while_while_cond_329445___redundant_placeholder24
0while_while_cond_329445___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?g
?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_325897

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
	ones_like\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:?????????&2
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????&2
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????&2
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????&2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:????????? 2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:????????? 2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:????????? 2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:????????? 2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:????????? 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:????????? 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:????????? 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:????????? 2
	BiasAdd_3e
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
mul_4e
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
mul_5e
mul_6Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
mul_6e
mul_7Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:????????? 2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:????????? 2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:????????? 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:????????? 2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
mul_10?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????&:????????? :????????? :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates
?0
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_327307
lstm_5_input
lstm_5_327271
lstm_5_327273
lstm_5_327275
dense_15_327278
dense_15_327280
dense_16_327284
dense_16_327286
dense_17_327289
dense_17_327291
identity?? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
lstm_5/StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputlstm_5_327271lstm_5_327273lstm_5_327275*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_3271142 
lstm_5/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_327278dense_15_327280*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_3271552"
 dense_15/StatefulPartitionedCall?
dropout_5/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3271882
dropout_5/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_16_327284dense_16_327286*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_3272122"
 dense_16/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_327289dense_17_327291*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_3272392"
 dense_17/StatefulPartitionedCall?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_327271*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_327273*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:Y U
+
_output_shapes
:?????????&
&
_user_specified_namelstm_5_input
??
?
while_body_329115
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resource??!while/lstm_cell_10/ReadVariableOp?#while/lstm_cell_10/ReadVariableOp_1?#while/lstm_cell_10/ReadVariableOp_2?#while/lstm_cell_10/ReadVariableOp_3?'while/lstm_cell_10/split/ReadVariableOp?)while/lstm_cell_10/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape?
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/ones_like/Const?
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/ones_like?
 while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/lstm_cell_10/dropout/Const?
while/lstm_cell_10/dropout/MulMul%while/lstm_cell_10/ones_like:output:0)while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????&2 
while/lstm_cell_10/dropout/Mul?
 while/lstm_cell_10/dropout/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_10/dropout/Shape?
7while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???29
7while/lstm_cell_10/dropout/random_uniform/RandomUniform?
)while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)while/lstm_cell_10/dropout/GreaterEqual/y?
'while/lstm_cell_10/dropout/GreaterEqualGreaterEqual@while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2)
'while/lstm_cell_10/dropout/GreaterEqual?
while/lstm_cell_10/dropout/CastCast+while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2!
while/lstm_cell_10/dropout/Cast?
 while/lstm_cell_10/dropout/Mul_1Mul"while/lstm_cell_10/dropout/Mul:z:0#while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout/Mul_1?
"while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_1/Const?
 while/lstm_cell_10/dropout_1/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout_1/Mul?
"while/lstm_cell_10/dropout_1/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_1/Shape?
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2??2;
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_1/GreaterEqual/y?
)while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2+
)while/lstm_cell_10/dropout_1/GreaterEqual?
!while/lstm_cell_10/dropout_1/CastCast-while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2#
!while/lstm_cell_10/dropout_1/Cast?
"while/lstm_cell_10/dropout_1/Mul_1Mul$while/lstm_cell_10/dropout_1/Mul:z:0%while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????&2$
"while/lstm_cell_10/dropout_1/Mul_1?
"while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_2/Const?
 while/lstm_cell_10/dropout_2/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout_2/Mul?
"while/lstm_cell_10/dropout_2/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_2/Shape?
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2;
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_2/GreaterEqual/y?
)while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2+
)while/lstm_cell_10/dropout_2/GreaterEqual?
!while/lstm_cell_10/dropout_2/CastCast-while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2#
!while/lstm_cell_10/dropout_2/Cast?
"while/lstm_cell_10/dropout_2/Mul_1Mul$while/lstm_cell_10/dropout_2/Mul:z:0%while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????&2$
"while/lstm_cell_10/dropout_2/Mul_1?
"while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_3/Const?
 while/lstm_cell_10/dropout_3/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout_3/Mul?
"while/lstm_cell_10/dropout_3/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_3/Shape?
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2;
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_3/GreaterEqual/y?
)while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2+
)while/lstm_cell_10/dropout_3/GreaterEqual?
!while/lstm_cell_10/dropout_3/CastCast-while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2#
!while/lstm_cell_10/dropout_3/Cast?
"while/lstm_cell_10/dropout_3/Mul_1Mul$while/lstm_cell_10/dropout_3/Mul:z:0%while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????&2$
"while/lstm_cell_10/dropout_3/Mul_1?
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape?
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$while/lstm_cell_10/ones_like_1/Const?
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2 
while/lstm_cell_10/ones_like_1?
"while/lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_4/Const?
 while/lstm_cell_10/dropout_4/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_4/Mul?
"while/lstm_cell_10/dropout_4/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_4/Shape?
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2Ѭ?2;
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_4/GreaterEqual/y?
)while/lstm_cell_10/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_4/GreaterEqual?
!while/lstm_cell_10/dropout_4/CastCast-while/lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_4/Cast?
"while/lstm_cell_10/dropout_4/Mul_1Mul$while/lstm_cell_10/dropout_4/Mul:z:0%while/lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_4/Mul_1?
"while/lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_5/Const?
 while/lstm_cell_10/dropout_5/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_5/Mul?
"while/lstm_cell_10/dropout_5/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_5/Shape?
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2ሳ2;
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_5/GreaterEqual/y?
)while/lstm_cell_10/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_5/GreaterEqual?
!while/lstm_cell_10/dropout_5/CastCast-while/lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_5/Cast?
"while/lstm_cell_10/dropout_5/Mul_1Mul$while/lstm_cell_10/dropout_5/Mul:z:0%while/lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_5/Mul_1?
"while/lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_6/Const?
 while/lstm_cell_10/dropout_6/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_6/Mul?
"while/lstm_cell_10/dropout_6/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_6/Shape?
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2;
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_6/GreaterEqual/y?
)while/lstm_cell_10/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_6/GreaterEqual?
!while/lstm_cell_10/dropout_6/CastCast-while/lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_6/Cast?
"while/lstm_cell_10/dropout_6/Mul_1Mul$while/lstm_cell_10/dropout_6/Mul:z:0%while/lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_6/Mul_1?
"while/lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_7/Const?
 while/lstm_cell_10/dropout_7/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_7/Mul?
"while/lstm_cell_10/dropout_7/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_7/Shape?
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2;
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_7/GreaterEqual/y?
)while/lstm_cell_10/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_7/GreaterEqual?
!while/lstm_cell_10/dropout_7/CastCast-while/lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_7/Cast?
"while/lstm_cell_10/dropout_7/Mul_1Mul$while/lstm_cell_10/dropout_7/Mul:z:0%while/lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_7/Mul_1?
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul?
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_1?
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_2?
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&?*
dtype02)
'while/lstm_cell_10/split/ReadVariableOp?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul?
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_2?
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1?
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim?
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOp?
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1?
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd?
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_1?
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_2?
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_3?
while/lstm_cell_10/mul_4Mulwhile_placeholder_2&while/lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_4?
while/lstm_cell_10/mul_5Mulwhile_placeholder_2&while/lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_5?
while/lstm_cell_10/mul_6Mulwhile_placeholder_2&while/lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_6?
while/lstm_cell_10/mul_7Mulwhile_placeholder_2&while/lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_7?
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!while/lstm_cell_10/ReadVariableOp?
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack?
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1?
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2?
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_slice?
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_4?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add?
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid?
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1?
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack?
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1?
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2?
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1?
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_5?
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_8?
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2?
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack?
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1?
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2?
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2?
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_6?
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_2?
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh?
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_9?
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_3?
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3?
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack?
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1?
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2?
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3?
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_7?
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_4?
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh_1?
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
??
?
while_body_328762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resource??!while/lstm_cell_10/ReadVariableOp?#while/lstm_cell_10/ReadVariableOp_1?#while/lstm_cell_10/ReadVariableOp_2?#while/lstm_cell_10/ReadVariableOp_3?'while/lstm_cell_10/split/ReadVariableOp?)while/lstm_cell_10/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape?
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/ones_like/Const?
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/ones_like?
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape?
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$while/lstm_cell_10/ones_like_1/Const?
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2 
while/lstm_cell_10/ones_like_1?
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul?
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_1?
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_2?
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&?*
dtype02)
'while/lstm_cell_10/split/ReadVariableOp?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul?
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_2?
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1?
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim?
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOp?
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1?
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd?
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_1?
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_2?
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_3?
while/lstm_cell_10/mul_4Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_4?
while/lstm_cell_10/mul_5Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_5?
while/lstm_cell_10/mul_6Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_6?
while/lstm_cell_10/mul_7Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_7?
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!while/lstm_cell_10/ReadVariableOp?
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack?
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1?
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2?
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_slice?
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_4?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add?
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid?
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1?
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack?
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1?
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2?
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1?
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_5?
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_8?
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2?
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack?
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1?
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2?
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2?
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_6?
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_2?
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh?
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_9?
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_3?
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3?
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack?
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1?
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2?
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3?
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_7?
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_4?
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh_1?
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?2
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_327268
lstm_5_input
lstm_5_327137
lstm_5_327139
lstm_5_327141
dense_15_327166
dense_15_327168
dense_16_327223
dense_16_327225
dense_17_327250
dense_17_327252
identity?? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
lstm_5/StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputlstm_5_327137lstm_5_327139lstm_5_327141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_3268472 
lstm_5/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_327166dense_15_327168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_3271552"
 dense_15/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3271832#
!dropout_5/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_16_327223dense_16_327225*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_3272122"
 dense_16/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_327250dense_17_327252*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_3272392"
 dense_17/StatefulPartitionedCall?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_327137*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_327139*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:Y U
+
_output_shapes
:?????????&
&
_user_specified_namelstm_5_input
??
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_329327

inputs.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identity??:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?lstm_cell_10/ReadVariableOp?lstm_cell_10/ReadVariableOp_1?lstm_cell_10/ReadVariableOp_2?lstm_cell_10/ReadVariableOp_3?!lstm_cell_10/split/ReadVariableOp?#lstm_cell_10/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2
strided_slice_2?
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape?
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/ones_like/Const?
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/ones_like}
lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout/Const?
lstm_cell_10/dropout/MulMullstm_cell_10/ones_like:output:0#lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout/Mul?
lstm_cell_10/dropout/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout/Shape?
1lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2?Ɔ23
1lstm_cell_10/dropout/random_uniform/RandomUniform?
#lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2%
#lstm_cell_10/dropout/GreaterEqual/y?
!lstm_cell_10/dropout/GreaterEqualGreaterEqual:lstm_cell_10/dropout/random_uniform/RandomUniform:output:0,lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2#
!lstm_cell_10/dropout/GreaterEqual?
lstm_cell_10/dropout/CastCast%lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout/Cast?
lstm_cell_10/dropout/Mul_1Mullstm_cell_10/dropout/Mul:z:0lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout/Mul_1?
lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_1/Const?
lstm_cell_10/dropout_1/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_1/Mul?
lstm_cell_10/dropout_1/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_1/Shape?
3lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_1/random_uniform/RandomUniform?
%lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_1/GreaterEqual/y?
#lstm_cell_10/dropout_1/GreaterEqualGreaterEqual<lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_cell_10/dropout_1/GreaterEqual?
lstm_cell_10/dropout_1/CastCast'lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_1/Cast?
lstm_cell_10/dropout_1/Mul_1Mullstm_cell_10/dropout_1/Mul:z:0lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_1/Mul_1?
lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_2/Const?
lstm_cell_10/dropout_2/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_2/Mul?
lstm_cell_10/dropout_2/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_2/Shape?
3lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_2/random_uniform/RandomUniform?
%lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_2/GreaterEqual/y?
#lstm_cell_10/dropout_2/GreaterEqualGreaterEqual<lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_cell_10/dropout_2/GreaterEqual?
lstm_cell_10/dropout_2/CastCast'lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_2/Cast?
lstm_cell_10/dropout_2/Mul_1Mullstm_cell_10/dropout_2/Mul:z:0lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_2/Mul_1?
lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_3/Const?
lstm_cell_10/dropout_3/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_3/Mul?
lstm_cell_10/dropout_3/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_3/Shape?
3lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2¬?25
3lstm_cell_10/dropout_3/random_uniform/RandomUniform?
%lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_3/GreaterEqual/y?
#lstm_cell_10/dropout_3/GreaterEqualGreaterEqual<lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_cell_10/dropout_3/GreaterEqual?
lstm_cell_10/dropout_3/CastCast'lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_3/Cast?
lstm_cell_10/dropout_3/Mul_1Mullstm_cell_10/dropout_3/Mul:z:0lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_3/Mul_1~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape?
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm_cell_10/ones_like_1/Const?
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/ones_like_1?
lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_4/Const?
lstm_cell_10/dropout_4/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_4/Mul?
lstm_cell_10/dropout_4/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_4/Shape?
3lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2궭25
3lstm_cell_10/dropout_4/random_uniform/RandomUniform?
%lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_4/GreaterEqual/y?
#lstm_cell_10/dropout_4/GreaterEqualGreaterEqual<lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_4/GreaterEqual?
lstm_cell_10/dropout_4/CastCast'lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_4/Cast?
lstm_cell_10/dropout_4/Mul_1Mullstm_cell_10/dropout_4/Mul:z:0lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_4/Mul_1?
lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_5/Const?
lstm_cell_10/dropout_5/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_5/Mul?
lstm_cell_10/dropout_5/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_5/Shape?
3lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_5/random_uniform/RandomUniform?
%lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_5/GreaterEqual/y?
#lstm_cell_10/dropout_5/GreaterEqualGreaterEqual<lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_5/GreaterEqual?
lstm_cell_10/dropout_5/CastCast'lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_5/Cast?
lstm_cell_10/dropout_5/Mul_1Mullstm_cell_10/dropout_5/Mul:z:0lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_5/Mul_1?
lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_6/Const?
lstm_cell_10/dropout_6/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_6/Mul?
lstm_cell_10/dropout_6/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_6/Shape?
3lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2?ʙ25
3lstm_cell_10/dropout_6/random_uniform/RandomUniform?
%lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_6/GreaterEqual/y?
#lstm_cell_10/dropout_6/GreaterEqualGreaterEqual<lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_6/GreaterEqual?
lstm_cell_10/dropout_6/CastCast'lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_6/Cast?
lstm_cell_10/dropout_6/Mul_1Mullstm_cell_10/dropout_6/Mul:z:0lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_6/Mul_1?
lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_7/Const?
lstm_cell_10/dropout_7/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_7/Mul?
lstm_cell_10/dropout_7/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_7/Shape?
3lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_7/random_uniform/RandomUniform?
%lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_7/GreaterEqual/y?
#lstm_cell_10/dropout_7/GreaterEqualGreaterEqual<lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_7/GreaterEqual?
lstm_cell_10/dropout_7/CastCast'lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_7/Cast?
lstm_cell_10/dropout_7/Mul_1Mullstm_cell_10/dropout_7/Mul:z:0lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_7/Mul_1?
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul?
lstm_cell_10/mul_1Mulstrided_slice_2:output:0 lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_1?
lstm_cell_10/mul_2Mulstrided_slice_2:output:0 lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_2?
lstm_cell_10/mul_3Mulstrided_slice_2:output:0 lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02#
!lstm_cell_10/split/ReadVariableOp?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split?
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul?
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_1?
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_2?
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1?
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dim?
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp?
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd?
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_1?
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_2?
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_3?
lstm_cell_10/mul_4Mulzeros:output:0 lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_4?
lstm_cell_10/mul_5Mulzeros:output:0 lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_5?
lstm_cell_10/mul_6Mulzeros:output:0 lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_6?
lstm_cell_10/mul_7Mulzeros:output:0 lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_7?
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp?
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack?
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1?
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2?
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice?
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_4?
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid?
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_1?
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack?
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1?
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2?
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1?
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_5?
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_8?
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_2?
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack?
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1?
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2?
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2?
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_6?
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh?
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_9?
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_3?
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_3?
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack?
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1?
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2?
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3?
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_7?
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_4?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh_1?
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_329115*
condR
while_cond_329114*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_330027G
Clstm_5_lstm_cell_10_bias_regularizer_square_readvariableop_resource
identity??:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpClstm_5_lstm_cell_10_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentity,lstm_5/lstm_cell_10/bias/Regularizer/mul:z:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp
?	
?
lstm_5_while_cond_328019*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1B
>lstm_5_while_lstm_5_while_cond_328019___redundant_placeholder0B
>lstm_5_while_lstm_5_while_cond_328019___redundant_placeholder1B
>lstm_5_while_lstm_5_while_cond_328019___redundant_placeholder2B
>lstm_5_while_lstm_5_while_cond_328019___redundant_placeholder3
lstm_5_while_identity
?
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
lstm_5/while/Lessr
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_5/while/Identity"7
lstm_5_while_identitylstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
??
?
while_body_329446
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resource??!while/lstm_cell_10/ReadVariableOp?#while/lstm_cell_10/ReadVariableOp_1?#while/lstm_cell_10/ReadVariableOp_2?#while/lstm_cell_10/ReadVariableOp_3?'while/lstm_cell_10/split/ReadVariableOp?)while/lstm_cell_10/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape?
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/ones_like/Const?
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/ones_like?
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape?
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$while/lstm_cell_10/ones_like_1/Const?
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2 
while/lstm_cell_10/ones_like_1?
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul?
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_1?
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_2?
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&?*
dtype02)
'while/lstm_cell_10/split/ReadVariableOp?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul?
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_2?
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1?
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim?
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOp?
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1?
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd?
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_1?
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_2?
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_3?
while/lstm_cell_10/mul_4Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_4?
while/lstm_cell_10/mul_5Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_5?
while/lstm_cell_10/mul_6Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_6?
while/lstm_cell_10/mul_7Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_7?
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!while/lstm_cell_10/ReadVariableOp?
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack?
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1?
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2?
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_slice?
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_4?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add?
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid?
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1?
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack?
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1?
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2?
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1?
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_5?
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_8?
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2?
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack?
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1?
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2?
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2?
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_6?
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_2?
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh?
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_9?
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_3?
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3?
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack?
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1?
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2?
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3?
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_7?
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_4?
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh_1?
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_329114
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_329114___redundant_placeholder04
0while_while_cond_329114___redundant_placeholder14
0while_while_cond_329114___redundant_placeholder24
0while_while_cond_329114___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
??
?
while_body_326635
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resource??!while/lstm_cell_10/ReadVariableOp?#while/lstm_cell_10/ReadVariableOp_1?#while/lstm_cell_10/ReadVariableOp_2?#while/lstm_cell_10/ReadVariableOp_3?'while/lstm_cell_10/split/ReadVariableOp?)while/lstm_cell_10/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape?
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/ones_like/Const?
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/ones_like?
 while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/lstm_cell_10/dropout/Const?
while/lstm_cell_10/dropout/MulMul%while/lstm_cell_10/ones_like:output:0)while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????&2 
while/lstm_cell_10/dropout/Mul?
 while/lstm_cell_10/dropout/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_10/dropout/Shape?
7while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???29
7while/lstm_cell_10/dropout/random_uniform/RandomUniform?
)while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)while/lstm_cell_10/dropout/GreaterEqual/y?
'while/lstm_cell_10/dropout/GreaterEqualGreaterEqual@while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2)
'while/lstm_cell_10/dropout/GreaterEqual?
while/lstm_cell_10/dropout/CastCast+while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2!
while/lstm_cell_10/dropout/Cast?
 while/lstm_cell_10/dropout/Mul_1Mul"while/lstm_cell_10/dropout/Mul:z:0#while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout/Mul_1?
"while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_1/Const?
 while/lstm_cell_10/dropout_1/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout_1/Mul?
"while/lstm_cell_10/dropout_1/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_1/Shape?
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2??92;
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_1/GreaterEqual/y?
)while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2+
)while/lstm_cell_10/dropout_1/GreaterEqual?
!while/lstm_cell_10/dropout_1/CastCast-while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2#
!while/lstm_cell_10/dropout_1/Cast?
"while/lstm_cell_10/dropout_1/Mul_1Mul$while/lstm_cell_10/dropout_1/Mul:z:0%while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????&2$
"while/lstm_cell_10/dropout_1/Mul_1?
"while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_2/Const?
 while/lstm_cell_10/dropout_2/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout_2/Mul?
"while/lstm_cell_10/dropout_2/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_2/Shape?
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2;
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_2/GreaterEqual/y?
)while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2+
)while/lstm_cell_10/dropout_2/GreaterEqual?
!while/lstm_cell_10/dropout_2/CastCast-while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2#
!while/lstm_cell_10/dropout_2/Cast?
"while/lstm_cell_10/dropout_2/Mul_1Mul$while/lstm_cell_10/dropout_2/Mul:z:0%while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????&2$
"while/lstm_cell_10/dropout_2/Mul_1?
"while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_3/Const?
 while/lstm_cell_10/dropout_3/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout_3/Mul?
"while/lstm_cell_10/dropout_3/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_3/Shape?
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2޶?2;
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_3/GreaterEqual/y?
)while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2+
)while/lstm_cell_10/dropout_3/GreaterEqual?
!while/lstm_cell_10/dropout_3/CastCast-while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2#
!while/lstm_cell_10/dropout_3/Cast?
"while/lstm_cell_10/dropout_3/Mul_1Mul$while/lstm_cell_10/dropout_3/Mul:z:0%while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????&2$
"while/lstm_cell_10/dropout_3/Mul_1?
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape?
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$while/lstm_cell_10/ones_like_1/Const?
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2 
while/lstm_cell_10/ones_like_1?
"while/lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_4/Const?
 while/lstm_cell_10/dropout_4/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_4/Mul?
"while/lstm_cell_10/dropout_4/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_4/Shape?
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2;
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_4/GreaterEqual/y?
)while/lstm_cell_10/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_4/GreaterEqual?
!while/lstm_cell_10/dropout_4/CastCast-while/lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_4/Cast?
"while/lstm_cell_10/dropout_4/Mul_1Mul$while/lstm_cell_10/dropout_4/Mul:z:0%while/lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_4/Mul_1?
"while/lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_5/Const?
 while/lstm_cell_10/dropout_5/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_5/Mul?
"while/lstm_cell_10/dropout_5/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_5/Shape?
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2?ي2;
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_5/GreaterEqual/y?
)while/lstm_cell_10/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_5/GreaterEqual?
!while/lstm_cell_10/dropout_5/CastCast-while/lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_5/Cast?
"while/lstm_cell_10/dropout_5/Mul_1Mul$while/lstm_cell_10/dropout_5/Mul:z:0%while/lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_5/Mul_1?
"while/lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_6/Const?
 while/lstm_cell_10/dropout_6/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_6/Mul?
"while/lstm_cell_10/dropout_6/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_6/Shape?
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2Ϊ?2;
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_6/GreaterEqual/y?
)while/lstm_cell_10/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_6/GreaterEqual?
!while/lstm_cell_10/dropout_6/CastCast-while/lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_6/Cast?
"while/lstm_cell_10/dropout_6/Mul_1Mul$while/lstm_cell_10/dropout_6/Mul:z:0%while/lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_6/Mul_1?
"while/lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_7/Const?
 while/lstm_cell_10/dropout_7/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_7/Mul?
"while/lstm_cell_10/dropout_7/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_7/Shape?
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2;
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_7/GreaterEqual/y?
)while/lstm_cell_10/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_7/GreaterEqual?
!while/lstm_cell_10/dropout_7/CastCast-while/lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_7/Cast?
"while/lstm_cell_10/dropout_7/Mul_1Mul$while/lstm_cell_10/dropout_7/Mul:z:0%while/lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_7/Mul_1?
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul?
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_1?
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_2?
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&?*
dtype02)
'while/lstm_cell_10/split/ReadVariableOp?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul?
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_2?
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1?
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim?
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOp?
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1?
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd?
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_1?
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_2?
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_3?
while/lstm_cell_10/mul_4Mulwhile_placeholder_2&while/lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_4?
while/lstm_cell_10/mul_5Mulwhile_placeholder_2&while/lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_5?
while/lstm_cell_10/mul_6Mulwhile_placeholder_2&while/lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_6?
while/lstm_cell_10/mul_7Mulwhile_placeholder_2&while/lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_7?
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!while/lstm_cell_10/ReadVariableOp?
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack?
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1?
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2?
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_slice?
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_4?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add?
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid?
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1?
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack?
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1?
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2?
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1?
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_5?
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_8?
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2?
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack?
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1?
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2?
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2?
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_6?
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_2?
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh?
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_9?
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_3?
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3?
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack?
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1?
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2?
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3?
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_7?
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_4?
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh_1?
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
??
?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_325801

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????&2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????&2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????&2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2??62(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????&2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????&2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2扃2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????&2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????&2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2ȅ?2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????&2
dropout_3/Mul_1\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/Const?
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_5/Const?
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_6/Const?
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shape?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2ӗ?2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_6/GreaterEqual/y?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_6/GreaterEqual?
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_6/Cast?
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_7/Const?
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shape?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2?ܑ2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_7/GreaterEqual/y?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_7/GreaterEqual?
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_7/Cast?
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:????????? 2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:????????? 2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:????????? 2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:????????? 2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:????????? 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:????????? 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:????????? 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:????????? 2
	BiasAdd_3d
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
mul_4d
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
mul_5d
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
mul_6d
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:????????? 2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:????????? 2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:????????? 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:????????? 2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
mul_10?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????&:????????? :????????? :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates
??
?

lstm_5_while_body_328020*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0?
;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_07
3lstm_5_while_lstm_cell_10_readvariableop_resource_0
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor;
7lstm_5_while_lstm_cell_10_split_readvariableop_resource=
9lstm_5_while_lstm_cell_10_split_1_readvariableop_resource5
1lstm_5_while_lstm_cell_10_readvariableop_resource??(lstm_5/while/lstm_cell_10/ReadVariableOp?*lstm_5/while/lstm_cell_10/ReadVariableOp_1?*lstm_5/while/lstm_cell_10/ReadVariableOp_2?*lstm_5/while/lstm_cell_10/ReadVariableOp_3?.lstm_5/while/lstm_cell_10/split/ReadVariableOp?0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp?
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   2@
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype022
0lstm_5/while/TensorArrayV2Read/TensorListGetItem?
)lstm_5/while/lstm_cell_10/ones_like/ShapeShape7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/ones_like/Shape?
)lstm_5/while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)lstm_5/while/lstm_cell_10/ones_like/Const?
#lstm_5/while/lstm_cell_10/ones_likeFill2lstm_5/while/lstm_cell_10/ones_like/Shape:output:02lstm_5/while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_5/while/lstm_cell_10/ones_like?
+lstm_5/while/lstm_cell_10/ones_like_1/ShapeShapelstm_5_while_placeholder_2*
T0*
_output_shapes
:2-
+lstm_5/while/lstm_cell_10/ones_like_1/Shape?
+lstm_5/while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+lstm_5/while/lstm_cell_10/ones_like_1/Const?
%lstm_5/while/lstm_cell_10/ones_like_1Fill4lstm_5/while/lstm_cell_10/ones_like_1/Shape:output:04lstm_5/while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2'
%lstm_5/while/lstm_cell_10/ones_like_1?
lstm_5/while/lstm_cell_10/mulMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_5/while/lstm_cell_10/mul?
lstm_5/while/lstm_cell_10/mul_1Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2!
lstm_5/while/lstm_cell_10/mul_1?
lstm_5/while/lstm_cell_10/mul_2Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2!
lstm_5/while/lstm_cell_10/mul_2?
lstm_5/while/lstm_cell_10/mul_3Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2!
lstm_5/while/lstm_cell_10/mul_3?
lstm_5/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
lstm_5/while/lstm_cell_10/Const?
)lstm_5/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)lstm_5/while/lstm_cell_10/split/split_dim?
.lstm_5/while/lstm_cell_10/split/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&?*
dtype020
.lstm_5/while/lstm_cell_10/split/ReadVariableOp?
lstm_5/while/lstm_cell_10/splitSplit2lstm_5/while/lstm_cell_10/split/split_dim:output:06lstm_5/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2!
lstm_5/while/lstm_cell_10/split?
 lstm_5/while/lstm_cell_10/MatMulMatMul!lstm_5/while/lstm_cell_10/mul:z:0(lstm_5/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2"
 lstm_5/while/lstm_cell_10/MatMul?
"lstm_5/while/lstm_cell_10/MatMul_1MatMul#lstm_5/while/lstm_cell_10/mul_1:z:0(lstm_5/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_1?
"lstm_5/while/lstm_cell_10/MatMul_2MatMul#lstm_5/while/lstm_cell_10/mul_2:z:0(lstm_5/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_2?
"lstm_5/while/lstm_cell_10/MatMul_3MatMul#lstm_5/while/lstm_cell_10/mul_3:z:0(lstm_5/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_3?
!lstm_5/while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2#
!lstm_5/while/lstm_cell_10/Const_1?
+lstm_5/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lstm_5/while/lstm_cell_10/split_1/split_dim?
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype022
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp?
!lstm_5/while/lstm_cell_10/split_1Split4lstm_5/while/lstm_cell_10/split_1/split_dim:output:08lstm_5/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2#
!lstm_5/while/lstm_cell_10/split_1?
!lstm_5/while/lstm_cell_10/BiasAddBiasAdd*lstm_5/while/lstm_cell_10/MatMul:product:0*lstm_5/while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2#
!lstm_5/while/lstm_cell_10/BiasAdd?
#lstm_5/while/lstm_cell_10/BiasAdd_1BiasAdd,lstm_5/while/lstm_cell_10/MatMul_1:product:0*lstm_5/while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2%
#lstm_5/while/lstm_cell_10/BiasAdd_1?
#lstm_5/while/lstm_cell_10/BiasAdd_2BiasAdd,lstm_5/while/lstm_cell_10/MatMul_2:product:0*lstm_5/while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2%
#lstm_5/while/lstm_cell_10/BiasAdd_2?
#lstm_5/while/lstm_cell_10/BiasAdd_3BiasAdd,lstm_5/while/lstm_cell_10/MatMul_3:product:0*lstm_5/while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2%
#lstm_5/while/lstm_cell_10/BiasAdd_3?
lstm_5/while/lstm_cell_10/mul_4Mullstm_5_while_placeholder_2.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_4?
lstm_5/while/lstm_cell_10/mul_5Mullstm_5_while_placeholder_2.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_5?
lstm_5/while/lstm_cell_10/mul_6Mullstm_5_while_placeholder_2.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_6?
lstm_5/while/lstm_cell_10/mul_7Mullstm_5_while_placeholder_2.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_7?
(lstm_5/while/lstm_cell_10/ReadVariableOpReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02*
(lstm_5/while/lstm_cell_10/ReadVariableOp?
-lstm_5/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-lstm_5/while/lstm_cell_10/strided_slice/stack?
/lstm_5/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/lstm_5/while/lstm_cell_10/strided_slice/stack_1?
/lstm_5/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/lstm_5/while/lstm_cell_10/strided_slice/stack_2?
'lstm_5/while/lstm_cell_10/strided_sliceStridedSlice0lstm_5/while/lstm_cell_10/ReadVariableOp:value:06lstm_5/while/lstm_cell_10/strided_slice/stack:output:08lstm_5/while/lstm_cell_10/strided_slice/stack_1:output:08lstm_5/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2)
'lstm_5/while/lstm_cell_10/strided_slice?
"lstm_5/while/lstm_cell_10/MatMul_4MatMul#lstm_5/while/lstm_cell_10/mul_4:z:00lstm_5/while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_4?
lstm_5/while/lstm_cell_10/addAddV2*lstm_5/while/lstm_cell_10/BiasAdd:output:0,lstm_5/while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
lstm_5/while/lstm_cell_10/add?
!lstm_5/while/lstm_cell_10/SigmoidSigmoid!lstm_5/while/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2#
!lstm_5/while/lstm_cell_10/Sigmoid?
*lstm_5/while/lstm_cell_10/ReadVariableOp_1ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_1?
/lstm_5/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/lstm_5/while/lstm_cell_10/strided_slice_1/stack?
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   23
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_1?
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_2?
)lstm_5/while/lstm_cell_10/strided_slice_1StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_1:value:08lstm_5/while/lstm_cell_10/strided_slice_1/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_1/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_1?
"lstm_5/while/lstm_cell_10/MatMul_5MatMul#lstm_5/while/lstm_cell_10/mul_5:z:02lstm_5/while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_5?
lstm_5/while/lstm_cell_10/add_1AddV2,lstm_5/while/lstm_cell_10/BiasAdd_1:output:0,lstm_5/while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/add_1?
#lstm_5/while/lstm_cell_10/Sigmoid_1Sigmoid#lstm_5/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2%
#lstm_5/while/lstm_cell_10/Sigmoid_1?
lstm_5/while/lstm_cell_10/mul_8Mul'lstm_5/while/lstm_cell_10/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_8?
*lstm_5/while/lstm_cell_10/ReadVariableOp_2ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_2?
/lstm_5/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   21
/lstm_5/while/lstm_cell_10/strided_slice_2/stack?
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   23
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_1?
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_2?
)lstm_5/while/lstm_cell_10/strided_slice_2StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_2:value:08lstm_5/while/lstm_cell_10/strided_slice_2/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_2/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_2?
"lstm_5/while/lstm_cell_10/MatMul_6MatMul#lstm_5/while/lstm_cell_10/mul_6:z:02lstm_5/while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_6?
lstm_5/while/lstm_cell_10/add_2AddV2,lstm_5/while/lstm_cell_10/BiasAdd_2:output:0,lstm_5/while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/add_2?
lstm_5/while/lstm_cell_10/TanhTanh#lstm_5/while/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2 
lstm_5/while/lstm_cell_10/Tanh?
lstm_5/while/lstm_cell_10/mul_9Mul%lstm_5/while/lstm_cell_10/Sigmoid:y:0"lstm_5/while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_9?
lstm_5/while/lstm_cell_10/add_3AddV2#lstm_5/while/lstm_cell_10/mul_8:z:0#lstm_5/while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/add_3?
*lstm_5/while/lstm_cell_10/ReadVariableOp_3ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_3?
/lstm_5/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   21
/lstm_5/while/lstm_cell_10/strided_slice_3/stack?
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_1?
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_2?
)lstm_5/while/lstm_cell_10/strided_slice_3StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_3:value:08lstm_5/while/lstm_cell_10/strided_slice_3/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_3/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_3?
"lstm_5/while/lstm_cell_10/MatMul_7MatMul#lstm_5/while/lstm_cell_10/mul_7:z:02lstm_5/while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_7?
lstm_5/while/lstm_cell_10/add_4AddV2,lstm_5/while/lstm_cell_10/BiasAdd_3:output:0,lstm_5/while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/add_4?
#lstm_5/while/lstm_cell_10/Sigmoid_2Sigmoid#lstm_5/while/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2%
#lstm_5/while/lstm_cell_10/Sigmoid_2?
 lstm_5/while/lstm_cell_10/Tanh_1Tanh#lstm_5/while/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2"
 lstm_5/while/lstm_cell_10/Tanh_1?
 lstm_5/while/lstm_cell_10/mul_10Mul'lstm_5/while/lstm_cell_10/Sigmoid_2:y:0$lstm_5/while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2"
 lstm_5/while/lstm_cell_10/mul_10?
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder$lstm_5/while/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype023
1lstm_5/while/TensorArrayV2Write/TensorListSetItemj
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add/y?
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/addn
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add_1/y?
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/add_1?
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity?
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_1?
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_2?
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_3?
lstm_5/while/Identity_4Identity$lstm_5/while/lstm_cell_10/mul_10:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
lstm_5/while/Identity_4?
lstm_5/while/Identity_5Identity#lstm_5/while/lstm_cell_10/add_3:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
lstm_5/while/Identity_5"7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"h
1lstm_5_while_lstm_cell_10_readvariableop_resource3lstm_5_while_lstm_cell_10_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_10_split_1_readvariableop_resource;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_10_split_readvariableop_resource9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0"?
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2T
(lstm_5/while/lstm_cell_10/ReadVariableOp(lstm_5/while/lstm_cell_10/ReadVariableOp2X
*lstm_5/while/lstm_cell_10/ReadVariableOp_1*lstm_5/while/lstm_cell_10/ReadVariableOp_12X
*lstm_5/while/lstm_cell_10/ReadVariableOp_2*lstm_5/while/lstm_cell_10/ReadVariableOp_22X
*lstm_5/while/lstm_cell_10/ReadVariableOp_3*lstm_5/while/lstm_cell_10/ReadVariableOp_32`
.lstm_5/while/lstm_cell_10/split/ReadVariableOp.lstm_5/while/lstm_cell_10/split/ReadVariableOp2d
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_signature_wrapper_327477
lstm_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_3256012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????&
&
_user_specified_namelstm_5_input
?
?
'__inference_lstm_5_layer_call_fn_328932
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_3264402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????&:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????&
"
_user_specified_name
inputs/0
?
F
*__inference_dropout_5_layer_call_fn_329663

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3271882
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
~
)__inference_dense_15_layer_call_fn_329636

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_3271552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_lstm_5_layer_call_fn_328921
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_3262962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????&:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????&
"
_user_specified_name
inputs/0
?1
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_327349

inputs
lstm_5_327313
lstm_5_327315
lstm_5_327317
dense_15_327320
dense_15_327322
dense_16_327326
dense_16_327328
dense_17_327331
dense_17_327333
identity?? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
lstm_5/StatefulPartitionedCallStatefulPartitionedCallinputslstm_5_327313lstm_5_327315lstm_5_327317*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_3268472 
lstm_5/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_327320dense_15_327322*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_3271552"
 dense_15/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3271832#
!dropout_5/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_16_327326dense_16_327328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_3272122"
 dense_16/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_327331dense_17_327333*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_3272392"
 dense_17/StatefulPartitionedCall?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_327313*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_327315*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
٘
?

lstm_5_while_body_327660*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0?
;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_07
3lstm_5_while_lstm_cell_10_readvariableop_resource_0
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor;
7lstm_5_while_lstm_cell_10_split_readvariableop_resource=
9lstm_5_while_lstm_cell_10_split_1_readvariableop_resource5
1lstm_5_while_lstm_cell_10_readvariableop_resource??(lstm_5/while/lstm_cell_10/ReadVariableOp?*lstm_5/while/lstm_cell_10/ReadVariableOp_1?*lstm_5/while/lstm_cell_10/ReadVariableOp_2?*lstm_5/while/lstm_cell_10/ReadVariableOp_3?.lstm_5/while/lstm_cell_10/split/ReadVariableOp?0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp?
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   2@
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype022
0lstm_5/while/TensorArrayV2Read/TensorListGetItem?
)lstm_5/while/lstm_cell_10/ones_like/ShapeShape7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/ones_like/Shape?
)lstm_5/while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)lstm_5/while/lstm_cell_10/ones_like/Const?
#lstm_5/while/lstm_cell_10/ones_likeFill2lstm_5/while/lstm_cell_10/ones_like/Shape:output:02lstm_5/while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_5/while/lstm_cell_10/ones_like?
'lstm_5/while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'lstm_5/while/lstm_cell_10/dropout/Const?
%lstm_5/while/lstm_cell_10/dropout/MulMul,lstm_5/while/lstm_cell_10/ones_like:output:00lstm_5/while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????&2'
%lstm_5/while/lstm_cell_10/dropout/Mul?
'lstm_5/while/lstm_cell_10/dropout/ShapeShape,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2)
'lstm_5/while/lstm_cell_10/dropout/Shape?
>lstm_5/while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform0lstm_5/while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2?ӆ2@
>lstm_5/while/lstm_cell_10/dropout/random_uniform/RandomUniform?
0lstm_5/while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>22
0lstm_5/while/lstm_cell_10/dropout/GreaterEqual/y?
.lstm_5/while/lstm_cell_10/dropout/GreaterEqualGreaterEqualGlstm_5/while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:09lstm_5/while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&20
.lstm_5/while/lstm_cell_10/dropout/GreaterEqual?
&lstm_5/while/lstm_cell_10/dropout/CastCast2lstm_5/while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2(
&lstm_5/while/lstm_cell_10/dropout/Cast?
'lstm_5/while/lstm_cell_10/dropout/Mul_1Mul)lstm_5/while/lstm_cell_10/dropout/Mul:z:0*lstm_5/while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????&2)
'lstm_5/while/lstm_cell_10/dropout/Mul_1?
)lstm_5/while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)lstm_5/while/lstm_cell_10/dropout_1/Const?
'lstm_5/while/lstm_cell_10/dropout_1/MulMul,lstm_5/while/lstm_cell_10/ones_like:output:02lstm_5/while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????&2)
'lstm_5/while/lstm_cell_10/dropout_1/Mul?
)lstm_5/while/lstm_cell_10/dropout_1/ShapeShape,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_1/Shape?
@lstm_5/while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2?ʠ2B
@lstm_5/while/lstm_cell_10/dropout_1/random_uniform/RandomUniform?
2lstm_5/while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>24
2lstm_5/while/lstm_cell_10/dropout_1/GreaterEqual/y?
0lstm_5/while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&22
0lstm_5/while/lstm_cell_10/dropout_1/GreaterEqual?
(lstm_5/while/lstm_cell_10/dropout_1/CastCast4lstm_5/while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2*
(lstm_5/while/lstm_cell_10/dropout_1/Cast?
)lstm_5/while/lstm_cell_10/dropout_1/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_1/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????&2+
)lstm_5/while/lstm_cell_10/dropout_1/Mul_1?
)lstm_5/while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)lstm_5/while/lstm_cell_10/dropout_2/Const?
'lstm_5/while/lstm_cell_10/dropout_2/MulMul,lstm_5/while/lstm_cell_10/ones_like:output:02lstm_5/while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????&2)
'lstm_5/while/lstm_cell_10/dropout_2/Mul?
)lstm_5/while/lstm_cell_10/dropout_2/ShapeShape,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_2/Shape?
@lstm_5/while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2㞈2B
@lstm_5/while/lstm_cell_10/dropout_2/random_uniform/RandomUniform?
2lstm_5/while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>24
2lstm_5/while/lstm_cell_10/dropout_2/GreaterEqual/y?
0lstm_5/while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&22
0lstm_5/while/lstm_cell_10/dropout_2/GreaterEqual?
(lstm_5/while/lstm_cell_10/dropout_2/CastCast4lstm_5/while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2*
(lstm_5/while/lstm_cell_10/dropout_2/Cast?
)lstm_5/while/lstm_cell_10/dropout_2/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_2/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????&2+
)lstm_5/while/lstm_cell_10/dropout_2/Mul_1?
)lstm_5/while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)lstm_5/while/lstm_cell_10/dropout_3/Const?
'lstm_5/while/lstm_cell_10/dropout_3/MulMul,lstm_5/while/lstm_cell_10/ones_like:output:02lstm_5/while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????&2)
'lstm_5/while/lstm_cell_10/dropout_3/Mul?
)lstm_5/while/lstm_cell_10/dropout_3/ShapeShape,lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_3/Shape?
@lstm_5/while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2B
@lstm_5/while/lstm_cell_10/dropout_3/random_uniform/RandomUniform?
2lstm_5/while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>24
2lstm_5/while/lstm_cell_10/dropout_3/GreaterEqual/y?
0lstm_5/while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&22
0lstm_5/while/lstm_cell_10/dropout_3/GreaterEqual?
(lstm_5/while/lstm_cell_10/dropout_3/CastCast4lstm_5/while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2*
(lstm_5/while/lstm_cell_10/dropout_3/Cast?
)lstm_5/while/lstm_cell_10/dropout_3/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_3/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????&2+
)lstm_5/while/lstm_cell_10/dropout_3/Mul_1?
+lstm_5/while/lstm_cell_10/ones_like_1/ShapeShapelstm_5_while_placeholder_2*
T0*
_output_shapes
:2-
+lstm_5/while/lstm_cell_10/ones_like_1/Shape?
+lstm_5/while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+lstm_5/while/lstm_cell_10/ones_like_1/Const?
%lstm_5/while/lstm_cell_10/ones_like_1Fill4lstm_5/while/lstm_cell_10/ones_like_1/Shape:output:04lstm_5/while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2'
%lstm_5/while/lstm_cell_10/ones_like_1?
)lstm_5/while/lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)lstm_5/while/lstm_cell_10/dropout_4/Const?
'lstm_5/while/lstm_cell_10/dropout_4/MulMul.lstm_5/while/lstm_cell_10/ones_like_1:output:02lstm_5/while/lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:????????? 2)
'lstm_5/while/lstm_cell_10/dropout_4/Mul?
)lstm_5/while/lstm_cell_10/dropout_4/ShapeShape.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_4/Shape?
@lstm_5/while/lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2ډ?2B
@lstm_5/while/lstm_cell_10/dropout_4/random_uniform/RandomUniform?
2lstm_5/while/lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>24
2lstm_5/while/lstm_cell_10/dropout_4/GreaterEqual/y?
0lstm_5/while/lstm_cell_10/dropout_4/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 22
0lstm_5/while/lstm_cell_10/dropout_4/GreaterEqual?
(lstm_5/while/lstm_cell_10/dropout_4/CastCast4lstm_5/while/lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2*
(lstm_5/while/lstm_cell_10/dropout_4/Cast?
)lstm_5/while/lstm_cell_10/dropout_4/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_4/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:????????? 2+
)lstm_5/while/lstm_cell_10/dropout_4/Mul_1?
)lstm_5/while/lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)lstm_5/while/lstm_cell_10/dropout_5/Const?
'lstm_5/while/lstm_cell_10/dropout_5/MulMul.lstm_5/while/lstm_cell_10/ones_like_1:output:02lstm_5/while/lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:????????? 2)
'lstm_5/while/lstm_cell_10/dropout_5/Mul?
)lstm_5/while/lstm_cell_10/dropout_5/ShapeShape.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_5/Shape?
@lstm_5/while/lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2⥳2B
@lstm_5/while/lstm_cell_10/dropout_5/random_uniform/RandomUniform?
2lstm_5/while/lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>24
2lstm_5/while/lstm_cell_10/dropout_5/GreaterEqual/y?
0lstm_5/while/lstm_cell_10/dropout_5/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 22
0lstm_5/while/lstm_cell_10/dropout_5/GreaterEqual?
(lstm_5/while/lstm_cell_10/dropout_5/CastCast4lstm_5/while/lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2*
(lstm_5/while/lstm_cell_10/dropout_5/Cast?
)lstm_5/while/lstm_cell_10/dropout_5/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_5/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:????????? 2+
)lstm_5/while/lstm_cell_10/dropout_5/Mul_1?
)lstm_5/while/lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)lstm_5/while/lstm_cell_10/dropout_6/Const?
'lstm_5/while/lstm_cell_10/dropout_6/MulMul.lstm_5/while/lstm_cell_10/ones_like_1:output:02lstm_5/while/lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:????????? 2)
'lstm_5/while/lstm_cell_10/dropout_6/Mul?
)lstm_5/while/lstm_cell_10/dropout_6/ShapeShape.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_6/Shape?
@lstm_5/while/lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??j2B
@lstm_5/while/lstm_cell_10/dropout_6/random_uniform/RandomUniform?
2lstm_5/while/lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>24
2lstm_5/while/lstm_cell_10/dropout_6/GreaterEqual/y?
0lstm_5/while/lstm_cell_10/dropout_6/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 22
0lstm_5/while/lstm_cell_10/dropout_6/GreaterEqual?
(lstm_5/while/lstm_cell_10/dropout_6/CastCast4lstm_5/while/lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2*
(lstm_5/while/lstm_cell_10/dropout_6/Cast?
)lstm_5/while/lstm_cell_10/dropout_6/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_6/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:????????? 2+
)lstm_5/while/lstm_cell_10/dropout_6/Mul_1?
)lstm_5/while/lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)lstm_5/while/lstm_cell_10/dropout_7/Const?
'lstm_5/while/lstm_cell_10/dropout_7/MulMul.lstm_5/while/lstm_cell_10/ones_like_1:output:02lstm_5/while/lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:????????? 2)
'lstm_5/while/lstm_cell_10/dropout_7/Mul?
)lstm_5/while/lstm_cell_10/dropout_7/ShapeShape.lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2+
)lstm_5/while/lstm_cell_10/dropout_7/Shape?
@lstm_5/while/lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform2lstm_5/while/lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed22B
@lstm_5/while/lstm_cell_10/dropout_7/random_uniform/RandomUniform?
2lstm_5/while/lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>24
2lstm_5/while/lstm_cell_10/dropout_7/GreaterEqual/y?
0lstm_5/while/lstm_cell_10/dropout_7/GreaterEqualGreaterEqualIlstm_5/while/lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:0;lstm_5/while/lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 22
0lstm_5/while/lstm_cell_10/dropout_7/GreaterEqual?
(lstm_5/while/lstm_cell_10/dropout_7/CastCast4lstm_5/while/lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2*
(lstm_5/while/lstm_cell_10/dropout_7/Cast?
)lstm_5/while/lstm_cell_10/dropout_7/Mul_1Mul+lstm_5/while/lstm_cell_10/dropout_7/Mul:z:0,lstm_5/while/lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:????????? 2+
)lstm_5/while/lstm_cell_10/dropout_7/Mul_1?
lstm_5/while/lstm_cell_10/mulMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_5/while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_5/while/lstm_cell_10/mul?
lstm_5/while/lstm_cell_10/mul_1Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_5/while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2!
lstm_5/while/lstm_cell_10/mul_1?
lstm_5/while/lstm_cell_10/mul_2Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_5/while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2!
lstm_5/while/lstm_cell_10/mul_2?
lstm_5/while/lstm_cell_10/mul_3Mul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_5/while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2!
lstm_5/while/lstm_cell_10/mul_3?
lstm_5/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
lstm_5/while/lstm_cell_10/Const?
)lstm_5/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)lstm_5/while/lstm_cell_10/split/split_dim?
.lstm_5/while/lstm_cell_10/split/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&?*
dtype020
.lstm_5/while/lstm_cell_10/split/ReadVariableOp?
lstm_5/while/lstm_cell_10/splitSplit2lstm_5/while/lstm_cell_10/split/split_dim:output:06lstm_5/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2!
lstm_5/while/lstm_cell_10/split?
 lstm_5/while/lstm_cell_10/MatMulMatMul!lstm_5/while/lstm_cell_10/mul:z:0(lstm_5/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2"
 lstm_5/while/lstm_cell_10/MatMul?
"lstm_5/while/lstm_cell_10/MatMul_1MatMul#lstm_5/while/lstm_cell_10/mul_1:z:0(lstm_5/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_1?
"lstm_5/while/lstm_cell_10/MatMul_2MatMul#lstm_5/while/lstm_cell_10/mul_2:z:0(lstm_5/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_2?
"lstm_5/while/lstm_cell_10/MatMul_3MatMul#lstm_5/while/lstm_cell_10/mul_3:z:0(lstm_5/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_3?
!lstm_5/while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2#
!lstm_5/while/lstm_cell_10/Const_1?
+lstm_5/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+lstm_5/while/lstm_cell_10/split_1/split_dim?
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype022
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp?
!lstm_5/while/lstm_cell_10/split_1Split4lstm_5/while/lstm_cell_10/split_1/split_dim:output:08lstm_5/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2#
!lstm_5/while/lstm_cell_10/split_1?
!lstm_5/while/lstm_cell_10/BiasAddBiasAdd*lstm_5/while/lstm_cell_10/MatMul:product:0*lstm_5/while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2#
!lstm_5/while/lstm_cell_10/BiasAdd?
#lstm_5/while/lstm_cell_10/BiasAdd_1BiasAdd,lstm_5/while/lstm_cell_10/MatMul_1:product:0*lstm_5/while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2%
#lstm_5/while/lstm_cell_10/BiasAdd_1?
#lstm_5/while/lstm_cell_10/BiasAdd_2BiasAdd,lstm_5/while/lstm_cell_10/MatMul_2:product:0*lstm_5/while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2%
#lstm_5/while/lstm_cell_10/BiasAdd_2?
#lstm_5/while/lstm_cell_10/BiasAdd_3BiasAdd,lstm_5/while/lstm_cell_10/MatMul_3:product:0*lstm_5/while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2%
#lstm_5/while/lstm_cell_10/BiasAdd_3?
lstm_5/while/lstm_cell_10/mul_4Mullstm_5_while_placeholder_2-lstm_5/while/lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_4?
lstm_5/while/lstm_cell_10/mul_5Mullstm_5_while_placeholder_2-lstm_5/while/lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_5?
lstm_5/while/lstm_cell_10/mul_6Mullstm_5_while_placeholder_2-lstm_5/while/lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_6?
lstm_5/while/lstm_cell_10/mul_7Mullstm_5_while_placeholder_2-lstm_5/while/lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_7?
(lstm_5/while/lstm_cell_10/ReadVariableOpReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02*
(lstm_5/while/lstm_cell_10/ReadVariableOp?
-lstm_5/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-lstm_5/while/lstm_cell_10/strided_slice/stack?
/lstm_5/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/lstm_5/while/lstm_cell_10/strided_slice/stack_1?
/lstm_5/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/lstm_5/while/lstm_cell_10/strided_slice/stack_2?
'lstm_5/while/lstm_cell_10/strided_sliceStridedSlice0lstm_5/while/lstm_cell_10/ReadVariableOp:value:06lstm_5/while/lstm_cell_10/strided_slice/stack:output:08lstm_5/while/lstm_cell_10/strided_slice/stack_1:output:08lstm_5/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2)
'lstm_5/while/lstm_cell_10/strided_slice?
"lstm_5/while/lstm_cell_10/MatMul_4MatMul#lstm_5/while/lstm_cell_10/mul_4:z:00lstm_5/while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_4?
lstm_5/while/lstm_cell_10/addAddV2*lstm_5/while/lstm_cell_10/BiasAdd:output:0,lstm_5/while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
lstm_5/while/lstm_cell_10/add?
!lstm_5/while/lstm_cell_10/SigmoidSigmoid!lstm_5/while/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2#
!lstm_5/while/lstm_cell_10/Sigmoid?
*lstm_5/while/lstm_cell_10/ReadVariableOp_1ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_1?
/lstm_5/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/lstm_5/while/lstm_cell_10/strided_slice_1/stack?
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   23
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_1?
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_1/stack_2?
)lstm_5/while/lstm_cell_10/strided_slice_1StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_1:value:08lstm_5/while/lstm_cell_10/strided_slice_1/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_1/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_1?
"lstm_5/while/lstm_cell_10/MatMul_5MatMul#lstm_5/while/lstm_cell_10/mul_5:z:02lstm_5/while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_5?
lstm_5/while/lstm_cell_10/add_1AddV2,lstm_5/while/lstm_cell_10/BiasAdd_1:output:0,lstm_5/while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/add_1?
#lstm_5/while/lstm_cell_10/Sigmoid_1Sigmoid#lstm_5/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2%
#lstm_5/while/lstm_cell_10/Sigmoid_1?
lstm_5/while/lstm_cell_10/mul_8Mul'lstm_5/while/lstm_cell_10/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_8?
*lstm_5/while/lstm_cell_10/ReadVariableOp_2ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_2?
/lstm_5/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   21
/lstm_5/while/lstm_cell_10/strided_slice_2/stack?
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   23
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_1?
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_2/stack_2?
)lstm_5/while/lstm_cell_10/strided_slice_2StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_2:value:08lstm_5/while/lstm_cell_10/strided_slice_2/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_2/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_2?
"lstm_5/while/lstm_cell_10/MatMul_6MatMul#lstm_5/while/lstm_cell_10/mul_6:z:02lstm_5/while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_6?
lstm_5/while/lstm_cell_10/add_2AddV2,lstm_5/while/lstm_cell_10/BiasAdd_2:output:0,lstm_5/while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/add_2?
lstm_5/while/lstm_cell_10/TanhTanh#lstm_5/while/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2 
lstm_5/while/lstm_cell_10/Tanh?
lstm_5/while/lstm_cell_10/mul_9Mul%lstm_5/while/lstm_cell_10/Sigmoid:y:0"lstm_5/while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/mul_9?
lstm_5/while/lstm_cell_10/add_3AddV2#lstm_5/while/lstm_cell_10/mul_8:z:0#lstm_5/while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/add_3?
*lstm_5/while/lstm_cell_10/ReadVariableOp_3ReadVariableOp3lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*lstm_5/while/lstm_cell_10/ReadVariableOp_3?
/lstm_5/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   21
/lstm_5/while/lstm_cell_10/strided_slice_3/stack?
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_1?
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1lstm_5/while/lstm_cell_10/strided_slice_3/stack_2?
)lstm_5/while/lstm_cell_10/strided_slice_3StridedSlice2lstm_5/while/lstm_cell_10/ReadVariableOp_3:value:08lstm_5/while/lstm_cell_10/strided_slice_3/stack:output:0:lstm_5/while/lstm_cell_10/strided_slice_3/stack_1:output:0:lstm_5/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2+
)lstm_5/while/lstm_cell_10/strided_slice_3?
"lstm_5/while/lstm_cell_10/MatMul_7MatMul#lstm_5/while/lstm_cell_10/mul_7:z:02lstm_5/while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2$
"lstm_5/while/lstm_cell_10/MatMul_7?
lstm_5/while/lstm_cell_10/add_4AddV2,lstm_5/while/lstm_cell_10/BiasAdd_3:output:0,lstm_5/while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/while/lstm_cell_10/add_4?
#lstm_5/while/lstm_cell_10/Sigmoid_2Sigmoid#lstm_5/while/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2%
#lstm_5/while/lstm_cell_10/Sigmoid_2?
 lstm_5/while/lstm_cell_10/Tanh_1Tanh#lstm_5/while/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2"
 lstm_5/while/lstm_cell_10/Tanh_1?
 lstm_5/while/lstm_cell_10/mul_10Mul'lstm_5/while/lstm_cell_10/Sigmoid_2:y:0$lstm_5/while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2"
 lstm_5/while/lstm_cell_10/mul_10?
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder$lstm_5/while/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype023
1lstm_5/while/TensorArrayV2Write/TensorListSetItemj
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add/y?
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/addn
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add_1/y?
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/add_1?
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity?
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_1?
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_2?
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_5/while/Identity_3?
lstm_5/while/Identity_4Identity$lstm_5/while/lstm_cell_10/mul_10:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
lstm_5/while/Identity_4?
lstm_5/while/Identity_5Identity#lstm_5/while/lstm_cell_10/add_3:z:0)^lstm_5/while/lstm_cell_10/ReadVariableOp+^lstm_5/while/lstm_cell_10/ReadVariableOp_1+^lstm_5/while/lstm_cell_10/ReadVariableOp_2+^lstm_5/while/lstm_cell_10/ReadVariableOp_3/^lstm_5/while/lstm_cell_10/split/ReadVariableOp1^lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
lstm_5/while/Identity_5"7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"h
1lstm_5_while_lstm_cell_10_readvariableop_resource3lstm_5_while_lstm_cell_10_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_10_split_1_readvariableop_resource;lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_10_split_readvariableop_resource9lstm_5_while_lstm_cell_10_split_readvariableop_resource_0"?
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2T
(lstm_5/while/lstm_cell_10/ReadVariableOp(lstm_5/while/lstm_cell_10/ReadVariableOp2X
*lstm_5/while/lstm_cell_10/ReadVariableOp_1*lstm_5/while/lstm_cell_10/ReadVariableOp_12X
*lstm_5/while/lstm_cell_10/ReadVariableOp_2*lstm_5/while/lstm_cell_10/ReadVariableOp_22X
*lstm_5/while/lstm_cell_10/ReadVariableOp_3*lstm_5/while/lstm_cell_10/ReadVariableOp_32`
.lstm_5/while/lstm_cell_10/split/ReadVariableOp.lstm_5/while/lstm_cell_10/split/ReadVariableOp2d
0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp0lstm_5/while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
??
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_326847

inputs.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identity??:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?lstm_cell_10/ReadVariableOp?lstm_cell_10/ReadVariableOp_1?lstm_cell_10/ReadVariableOp_2?lstm_cell_10/ReadVariableOp_3?!lstm_cell_10/split/ReadVariableOp?#lstm_cell_10/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2
strided_slice_2?
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape?
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/ones_like/Const?
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/ones_like}
lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout/Const?
lstm_cell_10/dropout/MulMullstm_cell_10/ones_like:output:0#lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout/Mul?
lstm_cell_10/dropout/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout/Shape?
1lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???23
1lstm_cell_10/dropout/random_uniform/RandomUniform?
#lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2%
#lstm_cell_10/dropout/GreaterEqual/y?
!lstm_cell_10/dropout/GreaterEqualGreaterEqual:lstm_cell_10/dropout/random_uniform/RandomUniform:output:0,lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2#
!lstm_cell_10/dropout/GreaterEqual?
lstm_cell_10/dropout/CastCast%lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout/Cast?
lstm_cell_10/dropout/Mul_1Mullstm_cell_10/dropout/Mul:z:0lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout/Mul_1?
lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_1/Const?
lstm_cell_10/dropout_1/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_1/Mul?
lstm_cell_10/dropout_1/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_1/Shape?
3lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2??k25
3lstm_cell_10/dropout_1/random_uniform/RandomUniform?
%lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_1/GreaterEqual/y?
#lstm_cell_10/dropout_1/GreaterEqualGreaterEqual<lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_cell_10/dropout_1/GreaterEqual?
lstm_cell_10/dropout_1/CastCast'lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_1/Cast?
lstm_cell_10/dropout_1/Mul_1Mullstm_cell_10/dropout_1/Mul:z:0lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_1/Mul_1?
lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_2/Const?
lstm_cell_10/dropout_2/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_2/Mul?
lstm_cell_10/dropout_2/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_2/Shape?
3lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_2/random_uniform/RandomUniform?
%lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_2/GreaterEqual/y?
#lstm_cell_10/dropout_2/GreaterEqualGreaterEqual<lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_cell_10/dropout_2/GreaterEqual?
lstm_cell_10/dropout_2/CastCast'lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_2/Cast?
lstm_cell_10/dropout_2/Mul_1Mullstm_cell_10/dropout_2/Mul:z:0lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_2/Mul_1?
lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_3/Const?
lstm_cell_10/dropout_3/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_3/Mul?
lstm_cell_10/dropout_3/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_3/Shape?
3lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2??V25
3lstm_cell_10/dropout_3/random_uniform/RandomUniform?
%lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_3/GreaterEqual/y?
#lstm_cell_10/dropout_3/GreaterEqualGreaterEqual<lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_cell_10/dropout_3/GreaterEqual?
lstm_cell_10/dropout_3/CastCast'lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_3/Cast?
lstm_cell_10/dropout_3/Mul_1Mullstm_cell_10/dropout_3/Mul:z:0lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_3/Mul_1~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape?
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm_cell_10/ones_like_1/Const?
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/ones_like_1?
lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_4/Const?
lstm_cell_10/dropout_4/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_4/Mul?
lstm_cell_10/dropout_4/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_4/Shape?
3lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2츏25
3lstm_cell_10/dropout_4/random_uniform/RandomUniform?
%lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_4/GreaterEqual/y?
#lstm_cell_10/dropout_4/GreaterEqualGreaterEqual<lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_4/GreaterEqual?
lstm_cell_10/dropout_4/CastCast'lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_4/Cast?
lstm_cell_10/dropout_4/Mul_1Mullstm_cell_10/dropout_4/Mul:z:0lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_4/Mul_1?
lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_5/Const?
lstm_cell_10/dropout_5/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_5/Mul?
lstm_cell_10/dropout_5/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_5/Shape?
3lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2ʗ?25
3lstm_cell_10/dropout_5/random_uniform/RandomUniform?
%lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_5/GreaterEqual/y?
#lstm_cell_10/dropout_5/GreaterEqualGreaterEqual<lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_5/GreaterEqual?
lstm_cell_10/dropout_5/CastCast'lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_5/Cast?
lstm_cell_10/dropout_5/Mul_1Mullstm_cell_10/dropout_5/Mul:z:0lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_5/Mul_1?
lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_6/Const?
lstm_cell_10/dropout_6/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_6/Mul?
lstm_cell_10/dropout_6/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_6/Shape?
3lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_6/random_uniform/RandomUniform?
%lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_6/GreaterEqual/y?
#lstm_cell_10/dropout_6/GreaterEqualGreaterEqual<lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_6/GreaterEqual?
lstm_cell_10/dropout_6/CastCast'lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_6/Cast?
lstm_cell_10/dropout_6/Mul_1Mullstm_cell_10/dropout_6/Mul:z:0lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_6/Mul_1?
lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_7/Const?
lstm_cell_10/dropout_7/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_7/Mul?
lstm_cell_10/dropout_7/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_7/Shape?
3lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_7/random_uniform/RandomUniform?
%lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_7/GreaterEqual/y?
#lstm_cell_10/dropout_7/GreaterEqualGreaterEqual<lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_7/GreaterEqual?
lstm_cell_10/dropout_7/CastCast'lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_7/Cast?
lstm_cell_10/dropout_7/Mul_1Mullstm_cell_10/dropout_7/Mul:z:0lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_7/Mul_1?
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul?
lstm_cell_10/mul_1Mulstrided_slice_2:output:0 lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_1?
lstm_cell_10/mul_2Mulstrided_slice_2:output:0 lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_2?
lstm_cell_10/mul_3Mulstrided_slice_2:output:0 lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02#
!lstm_cell_10/split/ReadVariableOp?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split?
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul?
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_1?
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_2?
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1?
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dim?
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp?
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd?
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_1?
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_2?
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_3?
lstm_cell_10/mul_4Mulzeros:output:0 lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_4?
lstm_cell_10/mul_5Mulzeros:output:0 lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_5?
lstm_cell_10/mul_6Mulzeros:output:0 lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_6?
lstm_cell_10/mul_7Mulzeros:output:0 lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_7?
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp?
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack?
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1?
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2?
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice?
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_4?
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid?
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_1?
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack?
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1?
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2?
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1?
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_5?
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_8?
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_2?
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack?
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1?
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2?
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2?
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_6?
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh?
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_9?
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_3?
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_3?
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack?
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1?
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2?
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3?
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_7?
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_4?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh_1?
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_326635*
condR
while_cond_326634*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
-__inference_sequential_5_layer_call_fn_328236

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_3274112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
-__inference_sequential_5_layer_call_fn_327432
lstm_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_3274112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????&
&
_user_specified_namelstm_5_input
?	
?
D__inference_dense_16_layer_call_and_return_conditional_losses_329674

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_lstm_5_layer_call_fn_329605

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_3268472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
??
?
while_body_326966
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resource??!while/lstm_cell_10/ReadVariableOp?#while/lstm_cell_10/ReadVariableOp_1?#while/lstm_cell_10/ReadVariableOp_2?#while/lstm_cell_10/ReadVariableOp_3?'while/lstm_cell_10/split/ReadVariableOp?)while/lstm_cell_10/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape?
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/ones_like/Const?
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/ones_like?
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape?
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$while/lstm_cell_10/ones_like_1/Const?
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2 
while/lstm_cell_10/ones_like_1?
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul?
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_1?
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_2?
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&?*
dtype02)
'while/lstm_cell_10/split/ReadVariableOp?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul?
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_2?
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1?
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim?
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOp?
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1?
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd?
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_1?
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_2?
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_3?
while/lstm_cell_10/mul_4Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_4?
while/lstm_cell_10/mul_5Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_5?
while/lstm_cell_10/mul_6Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_6?
while/lstm_cell_10/mul_7Mulwhile_placeholder_2'while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_7?
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!while/lstm_cell_10/ReadVariableOp?
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack?
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1?
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2?
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_slice?
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_4?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add?
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid?
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1?
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack?
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1?
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2?
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1?
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_5?
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_8?
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2?
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack?
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1?
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2?
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2?
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_6?
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_2?
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh?
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_9?
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_3?
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3?
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack?
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1?
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2?
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3?
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_7?
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_4?
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh_1?
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
޵
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_327114

inputs.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identity??:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?lstm_cell_10/ReadVariableOp?lstm_cell_10/ReadVariableOp_1?lstm_cell_10/ReadVariableOp_2?lstm_cell_10/ReadVariableOp_3?!lstm_cell_10/split/ReadVariableOp?#lstm_cell_10/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2
strided_slice_2?
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape?
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/ones_like/Const?
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/ones_like~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape?
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm_cell_10/ones_like_1/Const?
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/ones_like_1?
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul?
lstm_cell_10/mul_1Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_1?
lstm_cell_10/mul_2Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_2?
lstm_cell_10/mul_3Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02#
!lstm_cell_10/split/ReadVariableOp?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split?
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul?
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_1?
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_2?
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1?
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dim?
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp?
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd?
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_1?
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_2?
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_3?
lstm_cell_10/mul_4Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_4?
lstm_cell_10/mul_5Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_5?
lstm_cell_10/mul_6Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_6?
lstm_cell_10/mul_7Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_7?
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp?
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack?
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1?
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2?
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice?
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_4?
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid?
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_1?
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack?
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1?
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2?
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1?
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_5?
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_8?
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_2?
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack?
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1?
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2?
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2?
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_6?
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh?
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_9?
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_3?
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_3?
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack?
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1?
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2?
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3?
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_7?
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_4?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh_1?
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_326966*
condR
while_cond_326965*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
??
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_327901

inputs5
1lstm_5_lstm_cell_10_split_readvariableop_resource7
3lstm_5_lstm_cell_10_split_1_readvariableop_resource/
+lstm_5_lstm_cell_10_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity??dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?"lstm_5/lstm_cell_10/ReadVariableOp?$lstm_5/lstm_cell_10/ReadVariableOp_1?$lstm_5/lstm_cell_10/ReadVariableOp_2?$lstm_5/lstm_cell_10/ReadVariableOp_3?:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?(lstm_5/lstm_cell_10/split/ReadVariableOp?*lstm_5/lstm_cell_10/split_1/ReadVariableOp?lstm_5/whileR
lstm_5/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_5/Shape?
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack?
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1?
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2?
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros/mul/y?
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros/Less/y?
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros/packed/1?
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const?
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros_1/mul/y?
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros_1/Less/y?
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros_1/packed/1?
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const?
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/zeros_1?
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose/perm?
lstm_5/transpose	Transposeinputslstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????&2
lstm_5/transposed
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:2
lstm_5/Shape_1?
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_1/stack?
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_1?
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_2?
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slice_1?
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_5/TensorArrayV2/element_shape?
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2?
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   2>
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_5/TensorArrayUnstack/TensorListFromTensor?
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_2/stack?
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_1?
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_2?
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2
lstm_5/strided_slice_2?
#lstm_5/lstm_cell_10/ones_like/ShapeShapelstm_5/strided_slice_2:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/ones_like/Shape?
#lstm_5/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_5/lstm_cell_10/ones_like/Const?
lstm_5/lstm_cell_10/ones_likeFill,lstm_5/lstm_cell_10/ones_like/Shape:output:0,lstm_5/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_5/lstm_cell_10/ones_like?
!lstm_5/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!lstm_5/lstm_cell_10/dropout/Const?
lstm_5/lstm_cell_10/dropout/MulMul&lstm_5/lstm_cell_10/ones_like:output:0*lstm_5/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????&2!
lstm_5/lstm_cell_10/dropout/Mul?
!lstm_5/lstm_cell_10/dropout/ShapeShape&lstm_5/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2#
!lstm_5/lstm_cell_10/dropout/Shape?
8lstm_5/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform*lstm_5/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2:
8lstm_5/lstm_cell_10/dropout/random_uniform/RandomUniform?
*lstm_5/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2,
*lstm_5/lstm_cell_10/dropout/GreaterEqual/y?
(lstm_5/lstm_cell_10/dropout/GreaterEqualGreaterEqualAlstm_5/lstm_cell_10/dropout/random_uniform/RandomUniform:output:03lstm_5/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2*
(lstm_5/lstm_cell_10/dropout/GreaterEqual?
 lstm_5/lstm_cell_10/dropout/CastCast,lstm_5/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2"
 lstm_5/lstm_cell_10/dropout/Cast?
!lstm_5/lstm_cell_10/dropout/Mul_1Mul#lstm_5/lstm_cell_10/dropout/Mul:z:0$lstm_5/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????&2#
!lstm_5/lstm_cell_10/dropout/Mul_1?
#lstm_5/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_5/lstm_cell_10/dropout_1/Const?
!lstm_5/lstm_cell_10/dropout_1/MulMul&lstm_5/lstm_cell_10/ones_like:output:0,lstm_5/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????&2#
!lstm_5/lstm_cell_10/dropout_1/Mul?
#lstm_5/lstm_cell_10/dropout_1/ShapeShape&lstm_5/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_1/Shape?
:lstm_5/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2<
:lstm_5/lstm_cell_10/dropout_1/random_uniform/RandomUniform?
,lstm_5/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2.
,lstm_5/lstm_cell_10/dropout_1/GreaterEqual/y?
*lstm_5/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2,
*lstm_5/lstm_cell_10/dropout_1/GreaterEqual?
"lstm_5/lstm_cell_10/dropout_1/CastCast.lstm_5/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2$
"lstm_5/lstm_cell_10/dropout_1/Cast?
#lstm_5/lstm_cell_10/dropout_1/Mul_1Mul%lstm_5/lstm_cell_10/dropout_1/Mul:z:0&lstm_5/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????&2%
#lstm_5/lstm_cell_10/dropout_1/Mul_1?
#lstm_5/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_5/lstm_cell_10/dropout_2/Const?
!lstm_5/lstm_cell_10/dropout_2/MulMul&lstm_5/lstm_cell_10/ones_like:output:0,lstm_5/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????&2#
!lstm_5/lstm_cell_10/dropout_2/Mul?
#lstm_5/lstm_cell_10/dropout_2/ShapeShape&lstm_5/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_2/Shape?
:lstm_5/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2<
:lstm_5/lstm_cell_10/dropout_2/random_uniform/RandomUniform?
,lstm_5/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2.
,lstm_5/lstm_cell_10/dropout_2/GreaterEqual/y?
*lstm_5/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2,
*lstm_5/lstm_cell_10/dropout_2/GreaterEqual?
"lstm_5/lstm_cell_10/dropout_2/CastCast.lstm_5/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2$
"lstm_5/lstm_cell_10/dropout_2/Cast?
#lstm_5/lstm_cell_10/dropout_2/Mul_1Mul%lstm_5/lstm_cell_10/dropout_2/Mul:z:0&lstm_5/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????&2%
#lstm_5/lstm_cell_10/dropout_2/Mul_1?
#lstm_5/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_5/lstm_cell_10/dropout_3/Const?
!lstm_5/lstm_cell_10/dropout_3/MulMul&lstm_5/lstm_cell_10/ones_like:output:0,lstm_5/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????&2#
!lstm_5/lstm_cell_10/dropout_3/Mul?
#lstm_5/lstm_cell_10/dropout_3/ShapeShape&lstm_5/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_3/Shape?
:lstm_5/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2?ʓ2<
:lstm_5/lstm_cell_10/dropout_3/random_uniform/RandomUniform?
,lstm_5/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2.
,lstm_5/lstm_cell_10/dropout_3/GreaterEqual/y?
*lstm_5/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2,
*lstm_5/lstm_cell_10/dropout_3/GreaterEqual?
"lstm_5/lstm_cell_10/dropout_3/CastCast.lstm_5/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2$
"lstm_5/lstm_cell_10/dropout_3/Cast?
#lstm_5/lstm_cell_10/dropout_3/Mul_1Mul%lstm_5/lstm_cell_10/dropout_3/Mul:z:0&lstm_5/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????&2%
#lstm_5/lstm_cell_10/dropout_3/Mul_1?
%lstm_5/lstm_cell_10/ones_like_1/ShapeShapelstm_5/zeros:output:0*
T0*
_output_shapes
:2'
%lstm_5/lstm_cell_10/ones_like_1/Shape?
%lstm_5/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%lstm_5/lstm_cell_10/ones_like_1/Const?
lstm_5/lstm_cell_10/ones_like_1Fill.lstm_5/lstm_cell_10/ones_like_1/Shape:output:0.lstm_5/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/lstm_cell_10/ones_like_1?
#lstm_5/lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_5/lstm_cell_10/dropout_4/Const?
!lstm_5/lstm_cell_10/dropout_4/MulMul(lstm_5/lstm_cell_10/ones_like_1:output:0,lstm_5/lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:????????? 2#
!lstm_5/lstm_cell_10/dropout_4/Mul?
#lstm_5/lstm_cell_10/dropout_4/ShapeShape(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_4/Shape?
:lstm_5/lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2?ɬ2<
:lstm_5/lstm_cell_10/dropout_4/random_uniform/RandomUniform?
,lstm_5/lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2.
,lstm_5/lstm_cell_10/dropout_4/GreaterEqual/y?
*lstm_5/lstm_cell_10/dropout_4/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2,
*lstm_5/lstm_cell_10/dropout_4/GreaterEqual?
"lstm_5/lstm_cell_10/dropout_4/CastCast.lstm_5/lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2$
"lstm_5/lstm_cell_10/dropout_4/Cast?
#lstm_5/lstm_cell_10/dropout_4/Mul_1Mul%lstm_5/lstm_cell_10/dropout_4/Mul:z:0&lstm_5/lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:????????? 2%
#lstm_5/lstm_cell_10/dropout_4/Mul_1?
#lstm_5/lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_5/lstm_cell_10/dropout_5/Const?
!lstm_5/lstm_cell_10/dropout_5/MulMul(lstm_5/lstm_cell_10/ones_like_1:output:0,lstm_5/lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:????????? 2#
!lstm_5/lstm_cell_10/dropout_5/Mul?
#lstm_5/lstm_cell_10/dropout_5/ShapeShape(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_5/Shape?
:lstm_5/lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2<
:lstm_5/lstm_cell_10/dropout_5/random_uniform/RandomUniform?
,lstm_5/lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2.
,lstm_5/lstm_cell_10/dropout_5/GreaterEqual/y?
*lstm_5/lstm_cell_10/dropout_5/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2,
*lstm_5/lstm_cell_10/dropout_5/GreaterEqual?
"lstm_5/lstm_cell_10/dropout_5/CastCast.lstm_5/lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2$
"lstm_5/lstm_cell_10/dropout_5/Cast?
#lstm_5/lstm_cell_10/dropout_5/Mul_1Mul%lstm_5/lstm_cell_10/dropout_5/Mul:z:0&lstm_5/lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:????????? 2%
#lstm_5/lstm_cell_10/dropout_5/Mul_1?
#lstm_5/lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_5/lstm_cell_10/dropout_6/Const?
!lstm_5/lstm_cell_10/dropout_6/MulMul(lstm_5/lstm_cell_10/ones_like_1:output:0,lstm_5/lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:????????? 2#
!lstm_5/lstm_cell_10/dropout_6/Mul?
#lstm_5/lstm_cell_10/dropout_6/ShapeShape(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_6/Shape?
:lstm_5/lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2<
:lstm_5/lstm_cell_10/dropout_6/random_uniform/RandomUniform?
,lstm_5/lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2.
,lstm_5/lstm_cell_10/dropout_6/GreaterEqual/y?
*lstm_5/lstm_cell_10/dropout_6/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2,
*lstm_5/lstm_cell_10/dropout_6/GreaterEqual?
"lstm_5/lstm_cell_10/dropout_6/CastCast.lstm_5/lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2$
"lstm_5/lstm_cell_10/dropout_6/Cast?
#lstm_5/lstm_cell_10/dropout_6/Mul_1Mul%lstm_5/lstm_cell_10/dropout_6/Mul:z:0&lstm_5/lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:????????? 2%
#lstm_5/lstm_cell_10/dropout_6/Mul_1?
#lstm_5/lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_5/lstm_cell_10/dropout_7/Const?
!lstm_5/lstm_cell_10/dropout_7/MulMul(lstm_5/lstm_cell_10/ones_like_1:output:0,lstm_5/lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:????????? 2#
!lstm_5/lstm_cell_10/dropout_7/Mul?
#lstm_5/lstm_cell_10/dropout_7/ShapeShape(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/dropout_7/Shape?
:lstm_5/lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform,lstm_5/lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2?׉2<
:lstm_5/lstm_cell_10/dropout_7/random_uniform/RandomUniform?
,lstm_5/lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2.
,lstm_5/lstm_cell_10/dropout_7/GreaterEqual/y?
*lstm_5/lstm_cell_10/dropout_7/GreaterEqualGreaterEqualClstm_5/lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:05lstm_5/lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2,
*lstm_5/lstm_cell_10/dropout_7/GreaterEqual?
"lstm_5/lstm_cell_10/dropout_7/CastCast.lstm_5/lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2$
"lstm_5/lstm_cell_10/dropout_7/Cast?
#lstm_5/lstm_cell_10/dropout_7/Mul_1Mul%lstm_5/lstm_cell_10/dropout_7/Mul:z:0&lstm_5/lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:????????? 2%
#lstm_5/lstm_cell_10/dropout_7/Mul_1?
lstm_5/lstm_cell_10/mulMullstm_5/strided_slice_2:output:0%lstm_5/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_5/lstm_cell_10/mul?
lstm_5/lstm_cell_10/mul_1Mullstm_5/strided_slice_2:output:0'lstm_5/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_5/lstm_cell_10/mul_1?
lstm_5/lstm_cell_10/mul_2Mullstm_5/strided_slice_2:output:0'lstm_5/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_5/lstm_cell_10/mul_2?
lstm_5/lstm_cell_10/mul_3Mullstm_5/strided_slice_2:output:0'lstm_5/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_5/lstm_cell_10/mul_3x
lstm_5/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_10/Const?
#lstm_5/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_5/lstm_cell_10/split/split_dim?
(lstm_5/lstm_cell_10/split/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02*
(lstm_5/lstm_cell_10/split/ReadVariableOp?
lstm_5/lstm_cell_10/splitSplit,lstm_5/lstm_cell_10/split/split_dim:output:00lstm_5/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_5/lstm_cell_10/split?
lstm_5/lstm_cell_10/MatMulMatMullstm_5/lstm_cell_10/mul:z:0"lstm_5/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul?
lstm_5/lstm_cell_10/MatMul_1MatMullstm_5/lstm_cell_10/mul_1:z:0"lstm_5/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_1?
lstm_5/lstm_cell_10/MatMul_2MatMullstm_5/lstm_cell_10/mul_2:z:0"lstm_5/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_2?
lstm_5/lstm_cell_10/MatMul_3MatMullstm_5/lstm_cell_10/mul_3:z:0"lstm_5/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_3|
lstm_5/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_10/Const_1?
%lstm_5/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%lstm_5/lstm_cell_10/split_1/split_dim?
*lstm_5/lstm_cell_10/split_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*lstm_5/lstm_cell_10/split_1/ReadVariableOp?
lstm_5/lstm_cell_10/split_1Split.lstm_5/lstm_cell_10/split_1/split_dim:output:02lstm_5/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_5/lstm_cell_10/split_1?
lstm_5/lstm_cell_10/BiasAddBiasAdd$lstm_5/lstm_cell_10/MatMul:product:0$lstm_5/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/BiasAdd?
lstm_5/lstm_cell_10/BiasAdd_1BiasAdd&lstm_5/lstm_cell_10/MatMul_1:product:0$lstm_5/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/BiasAdd_1?
lstm_5/lstm_cell_10/BiasAdd_2BiasAdd&lstm_5/lstm_cell_10/MatMul_2:product:0$lstm_5/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/BiasAdd_2?
lstm_5/lstm_cell_10/BiasAdd_3BiasAdd&lstm_5/lstm_cell_10/MatMul_3:product:0$lstm_5/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/BiasAdd_3?
lstm_5/lstm_cell_10/mul_4Mullstm_5/zeros:output:0'lstm_5/lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_4?
lstm_5/lstm_cell_10/mul_5Mullstm_5/zeros:output:0'lstm_5/lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_5?
lstm_5/lstm_cell_10/mul_6Mullstm_5/zeros:output:0'lstm_5/lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_6?
lstm_5/lstm_cell_10/mul_7Mullstm_5/zeros:output:0'lstm_5/lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_7?
"lstm_5/lstm_cell_10/ReadVariableOpReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"lstm_5/lstm_cell_10/ReadVariableOp?
'lstm_5/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'lstm_5/lstm_cell_10/strided_slice/stack?
)lstm_5/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_5/lstm_cell_10/strided_slice/stack_1?
)lstm_5/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)lstm_5/lstm_cell_10/strided_slice/stack_2?
!lstm_5/lstm_cell_10/strided_sliceStridedSlice*lstm_5/lstm_cell_10/ReadVariableOp:value:00lstm_5/lstm_cell_10/strided_slice/stack:output:02lstm_5/lstm_cell_10/strided_slice/stack_1:output:02lstm_5/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!lstm_5/lstm_cell_10/strided_slice?
lstm_5/lstm_cell_10/MatMul_4MatMullstm_5/lstm_cell_10/mul_4:z:0*lstm_5/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_4?
lstm_5/lstm_cell_10/addAddV2$lstm_5/lstm_cell_10/BiasAdd:output:0&lstm_5/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/add?
lstm_5/lstm_cell_10/SigmoidSigmoidlstm_5/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/Sigmoid?
$lstm_5/lstm_cell_10/ReadVariableOp_1ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_1?
)lstm_5/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_5/lstm_cell_10/strided_slice_1/stack?
+lstm_5/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+lstm_5/lstm_cell_10/strided_slice_1/stack_1?
+lstm_5/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_1/stack_2?
#lstm_5/lstm_cell_10/strided_slice_1StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_1:value:02lstm_5/lstm_cell_10/strided_slice_1/stack:output:04lstm_5/lstm_cell_10/strided_slice_1/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_1?
lstm_5/lstm_cell_10/MatMul_5MatMullstm_5/lstm_cell_10/mul_5:z:0,lstm_5/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_5?
lstm_5/lstm_cell_10/add_1AddV2&lstm_5/lstm_cell_10/BiasAdd_1:output:0&lstm_5/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/add_1?
lstm_5/lstm_cell_10/Sigmoid_1Sigmoidlstm_5/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/Sigmoid_1?
lstm_5/lstm_cell_10/mul_8Mul!lstm_5/lstm_cell_10/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_8?
$lstm_5/lstm_cell_10/ReadVariableOp_2ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_2?
)lstm_5/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)lstm_5/lstm_cell_10/strided_slice_2/stack?
+lstm_5/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+lstm_5/lstm_cell_10/strided_slice_2/stack_1?
+lstm_5/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_2/stack_2?
#lstm_5/lstm_cell_10/strided_slice_2StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_2:value:02lstm_5/lstm_cell_10/strided_slice_2/stack:output:04lstm_5/lstm_cell_10/strided_slice_2/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_2?
lstm_5/lstm_cell_10/MatMul_6MatMullstm_5/lstm_cell_10/mul_6:z:0,lstm_5/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_6?
lstm_5/lstm_cell_10/add_2AddV2&lstm_5/lstm_cell_10/BiasAdd_2:output:0&lstm_5/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/add_2?
lstm_5/lstm_cell_10/TanhTanhlstm_5/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/Tanh?
lstm_5/lstm_cell_10/mul_9Mullstm_5/lstm_cell_10/Sigmoid:y:0lstm_5/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_9?
lstm_5/lstm_cell_10/add_3AddV2lstm_5/lstm_cell_10/mul_8:z:0lstm_5/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/add_3?
$lstm_5/lstm_cell_10/ReadVariableOp_3ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_3?
)lstm_5/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)lstm_5/lstm_cell_10/strided_slice_3/stack?
+lstm_5/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_5/lstm_cell_10/strided_slice_3/stack_1?
+lstm_5/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_3/stack_2?
#lstm_5/lstm_cell_10/strided_slice_3StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_3:value:02lstm_5/lstm_cell_10/strided_slice_3/stack:output:04lstm_5/lstm_cell_10/strided_slice_3/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_3?
lstm_5/lstm_cell_10/MatMul_7MatMullstm_5/lstm_cell_10/mul_7:z:0,lstm_5/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_7?
lstm_5/lstm_cell_10/add_4AddV2&lstm_5/lstm_cell_10/BiasAdd_3:output:0&lstm_5/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/add_4?
lstm_5/lstm_cell_10/Sigmoid_2Sigmoidlstm_5/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/Sigmoid_2?
lstm_5/lstm_cell_10/Tanh_1Tanhlstm_5/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/Tanh_1?
lstm_5/lstm_cell_10/mul_10Mul!lstm_5/lstm_cell_10/Sigmoid_2:y:0lstm_5/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_10?
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2&
$lstm_5/TensorArrayV2_1/element_shape?
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2_1\
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/time?
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_5/while/maximum_iterationsx
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/while/loop_counter?
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_10_split_readvariableop_resource3lstm_5_lstm_cell_10_split_1_readvariableop_resource+lstm_5_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_5_while_body_327660*$
condR
lstm_5_while_cond_327659*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
lstm_5/while?
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02+
)lstm_5/TensorArrayV2Stack/TensorListStack?
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_5/strided_slice_3/stack?
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_5/strided_slice_3/stack_1?
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_3/stack_2?
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
lstm_5/strided_slice_3?
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose_1/perm?
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
lstm_5/transpose_1t
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/runtime?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMullstm_5/strided_slice_3:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/BiasAdd|
dense_15/SigmoidSigmoiddense_15/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_15/Sigmoidw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMuldense_15/Sigmoid:y:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_5/dropout/Mulv
dropout_5/dropout/ShapeShapedense_15/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_5/dropout/Mul_1?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMuldropout_5/dropout/Mul_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_16/BiasAdd|
dense_16/SigmoidSigmoiddense_16/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_16/Sigmoid?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_16/Sigmoid:y:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Softmax?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentitydense_17/Softmax:softmax:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp#^lstm_5/lstm_cell_10/ReadVariableOp%^lstm_5/lstm_cell_10/ReadVariableOp_1%^lstm_5/lstm_cell_10/ReadVariableOp_2%^lstm_5/lstm_cell_10/ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp)^lstm_5/lstm_cell_10/split/ReadVariableOp+^lstm_5/lstm_cell_10/split_1/ReadVariableOp^lstm_5/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2H
"lstm_5/lstm_cell_10/ReadVariableOp"lstm_5/lstm_cell_10/ReadVariableOp2L
$lstm_5/lstm_cell_10/ReadVariableOp_1$lstm_5/lstm_cell_10/ReadVariableOp_12L
$lstm_5/lstm_cell_10/ReadVariableOp_2$lstm_5/lstm_cell_10/ReadVariableOp_22L
$lstm_5/lstm_cell_10/ReadVariableOp_3$lstm_5/lstm_cell_10/ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2T
(lstm_5/lstm_cell_10/split/ReadVariableOp(lstm_5/lstm_cell_10/split/ReadVariableOp2X
*lstm_5/lstm_cell_10/split_1/ReadVariableOp*lstm_5/lstm_cell_10/split_1/ReadVariableOp2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
??
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_328190

inputs5
1lstm_5_lstm_cell_10_split_readvariableop_resource7
3lstm_5_lstm_cell_10_split_1_readvariableop_resource/
+lstm_5_lstm_cell_10_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity??dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?"lstm_5/lstm_cell_10/ReadVariableOp?$lstm_5/lstm_cell_10/ReadVariableOp_1?$lstm_5/lstm_cell_10/ReadVariableOp_2?$lstm_5/lstm_cell_10/ReadVariableOp_3?:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?(lstm_5/lstm_cell_10/split/ReadVariableOp?*lstm_5/lstm_cell_10/split_1/ReadVariableOp?lstm_5/whileR
lstm_5/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_5/Shape?
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack?
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1?
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2?
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros/mul/y?
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros/Less/y?
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros/packed/1?
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const?
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros_1/mul/y?
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros_1/Less/y?
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/zeros_1/packed/1?
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const?
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/zeros_1?
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose/perm?
lstm_5/transpose	Transposeinputslstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????&2
lstm_5/transposed
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:2
lstm_5/Shape_1?
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_1/stack?
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_1?
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_2?
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slice_1?
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_5/TensorArrayV2/element_shape?
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2?
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   2>
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_5/TensorArrayUnstack/TensorListFromTensor?
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_2/stack?
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_1?
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_2?
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2
lstm_5/strided_slice_2?
#lstm_5/lstm_cell_10/ones_like/ShapeShapelstm_5/strided_slice_2:output:0*
T0*
_output_shapes
:2%
#lstm_5/lstm_cell_10/ones_like/Shape?
#lstm_5/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_5/lstm_cell_10/ones_like/Const?
lstm_5/lstm_cell_10/ones_likeFill,lstm_5/lstm_cell_10/ones_like/Shape:output:0,lstm_5/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_5/lstm_cell_10/ones_like?
%lstm_5/lstm_cell_10/ones_like_1/ShapeShapelstm_5/zeros:output:0*
T0*
_output_shapes
:2'
%lstm_5/lstm_cell_10/ones_like_1/Shape?
%lstm_5/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%lstm_5/lstm_cell_10/ones_like_1/Const?
lstm_5/lstm_cell_10/ones_like_1Fill.lstm_5/lstm_cell_10/ones_like_1/Shape:output:0.lstm_5/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2!
lstm_5/lstm_cell_10/ones_like_1?
lstm_5/lstm_cell_10/mulMullstm_5/strided_slice_2:output:0&lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_5/lstm_cell_10/mul?
lstm_5/lstm_cell_10/mul_1Mullstm_5/strided_slice_2:output:0&lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_5/lstm_cell_10/mul_1?
lstm_5/lstm_cell_10/mul_2Mullstm_5/strided_slice_2:output:0&lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_5/lstm_cell_10/mul_2?
lstm_5/lstm_cell_10/mul_3Mullstm_5/strided_slice_2:output:0&lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_5/lstm_cell_10/mul_3x
lstm_5/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_10/Const?
#lstm_5/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_5/lstm_cell_10/split/split_dim?
(lstm_5/lstm_cell_10/split/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02*
(lstm_5/lstm_cell_10/split/ReadVariableOp?
lstm_5/lstm_cell_10/splitSplit,lstm_5/lstm_cell_10/split/split_dim:output:00lstm_5/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_5/lstm_cell_10/split?
lstm_5/lstm_cell_10/MatMulMatMullstm_5/lstm_cell_10/mul:z:0"lstm_5/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul?
lstm_5/lstm_cell_10/MatMul_1MatMullstm_5/lstm_cell_10/mul_1:z:0"lstm_5/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_1?
lstm_5/lstm_cell_10/MatMul_2MatMullstm_5/lstm_cell_10/mul_2:z:0"lstm_5/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_2?
lstm_5/lstm_cell_10/MatMul_3MatMullstm_5/lstm_cell_10/mul_3:z:0"lstm_5/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_3|
lstm_5/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_10/Const_1?
%lstm_5/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%lstm_5/lstm_cell_10/split_1/split_dim?
*lstm_5/lstm_cell_10/split_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*lstm_5/lstm_cell_10/split_1/ReadVariableOp?
lstm_5/lstm_cell_10/split_1Split.lstm_5/lstm_cell_10/split_1/split_dim:output:02lstm_5/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_5/lstm_cell_10/split_1?
lstm_5/lstm_cell_10/BiasAddBiasAdd$lstm_5/lstm_cell_10/MatMul:product:0$lstm_5/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/BiasAdd?
lstm_5/lstm_cell_10/BiasAdd_1BiasAdd&lstm_5/lstm_cell_10/MatMul_1:product:0$lstm_5/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/BiasAdd_1?
lstm_5/lstm_cell_10/BiasAdd_2BiasAdd&lstm_5/lstm_cell_10/MatMul_2:product:0$lstm_5/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/BiasAdd_2?
lstm_5/lstm_cell_10/BiasAdd_3BiasAdd&lstm_5/lstm_cell_10/MatMul_3:product:0$lstm_5/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/BiasAdd_3?
lstm_5/lstm_cell_10/mul_4Mullstm_5/zeros:output:0(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_4?
lstm_5/lstm_cell_10/mul_5Mullstm_5/zeros:output:0(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_5?
lstm_5/lstm_cell_10/mul_6Mullstm_5/zeros:output:0(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_6?
lstm_5/lstm_cell_10/mul_7Mullstm_5/zeros:output:0(lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_7?
"lstm_5/lstm_cell_10/ReadVariableOpReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"lstm_5/lstm_cell_10/ReadVariableOp?
'lstm_5/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'lstm_5/lstm_cell_10/strided_slice/stack?
)lstm_5/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_5/lstm_cell_10/strided_slice/stack_1?
)lstm_5/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)lstm_5/lstm_cell_10/strided_slice/stack_2?
!lstm_5/lstm_cell_10/strided_sliceStridedSlice*lstm_5/lstm_cell_10/ReadVariableOp:value:00lstm_5/lstm_cell_10/strided_slice/stack:output:02lstm_5/lstm_cell_10/strided_slice/stack_1:output:02lstm_5/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2#
!lstm_5/lstm_cell_10/strided_slice?
lstm_5/lstm_cell_10/MatMul_4MatMullstm_5/lstm_cell_10/mul_4:z:0*lstm_5/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_4?
lstm_5/lstm_cell_10/addAddV2$lstm_5/lstm_cell_10/BiasAdd:output:0&lstm_5/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/add?
lstm_5/lstm_cell_10/SigmoidSigmoidlstm_5/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/Sigmoid?
$lstm_5/lstm_cell_10/ReadVariableOp_1ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_1?
)lstm_5/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_5/lstm_cell_10/strided_slice_1/stack?
+lstm_5/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+lstm_5/lstm_cell_10/strided_slice_1/stack_1?
+lstm_5/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_1/stack_2?
#lstm_5/lstm_cell_10/strided_slice_1StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_1:value:02lstm_5/lstm_cell_10/strided_slice_1/stack:output:04lstm_5/lstm_cell_10/strided_slice_1/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_1?
lstm_5/lstm_cell_10/MatMul_5MatMullstm_5/lstm_cell_10/mul_5:z:0,lstm_5/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_5?
lstm_5/lstm_cell_10/add_1AddV2&lstm_5/lstm_cell_10/BiasAdd_1:output:0&lstm_5/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/add_1?
lstm_5/lstm_cell_10/Sigmoid_1Sigmoidlstm_5/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/Sigmoid_1?
lstm_5/lstm_cell_10/mul_8Mul!lstm_5/lstm_cell_10/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_8?
$lstm_5/lstm_cell_10/ReadVariableOp_2ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_2?
)lstm_5/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)lstm_5/lstm_cell_10/strided_slice_2/stack?
+lstm_5/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2-
+lstm_5/lstm_cell_10/strided_slice_2/stack_1?
+lstm_5/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_2/stack_2?
#lstm_5/lstm_cell_10/strided_slice_2StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_2:value:02lstm_5/lstm_cell_10/strided_slice_2/stack:output:04lstm_5/lstm_cell_10/strided_slice_2/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_2?
lstm_5/lstm_cell_10/MatMul_6MatMullstm_5/lstm_cell_10/mul_6:z:0,lstm_5/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_6?
lstm_5/lstm_cell_10/add_2AddV2&lstm_5/lstm_cell_10/BiasAdd_2:output:0&lstm_5/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/add_2?
lstm_5/lstm_cell_10/TanhTanhlstm_5/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/Tanh?
lstm_5/lstm_cell_10/mul_9Mullstm_5/lstm_cell_10/Sigmoid:y:0lstm_5/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_9?
lstm_5/lstm_cell_10/add_3AddV2lstm_5/lstm_cell_10/mul_8:z:0lstm_5/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/add_3?
$lstm_5/lstm_cell_10/ReadVariableOp_3ReadVariableOp+lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_5/lstm_cell_10/ReadVariableOp_3?
)lstm_5/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2+
)lstm_5/lstm_cell_10/strided_slice_3/stack?
+lstm_5/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_5/lstm_cell_10/strided_slice_3/stack_1?
+lstm_5/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_5/lstm_cell_10/strided_slice_3/stack_2?
#lstm_5/lstm_cell_10/strided_slice_3StridedSlice,lstm_5/lstm_cell_10/ReadVariableOp_3:value:02lstm_5/lstm_cell_10/strided_slice_3/stack:output:04lstm_5/lstm_cell_10/strided_slice_3/stack_1:output:04lstm_5/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2%
#lstm_5/lstm_cell_10/strided_slice_3?
lstm_5/lstm_cell_10/MatMul_7MatMullstm_5/lstm_cell_10/mul_7:z:0,lstm_5/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/MatMul_7?
lstm_5/lstm_cell_10/add_4AddV2&lstm_5/lstm_cell_10/BiasAdd_3:output:0&lstm_5/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/add_4?
lstm_5/lstm_cell_10/Sigmoid_2Sigmoidlstm_5/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/Sigmoid_2?
lstm_5/lstm_cell_10/Tanh_1Tanhlstm_5/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/Tanh_1?
lstm_5/lstm_cell_10/mul_10Mul!lstm_5/lstm_cell_10/Sigmoid_2:y:0lstm_5/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
lstm_5/lstm_cell_10/mul_10?
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2&
$lstm_5/TensorArrayV2_1/element_shape?
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2_1\
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/time?
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_5/while/maximum_iterationsx
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/while/loop_counter?
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_10_split_readvariableop_resource3lstm_5_lstm_cell_10_split_1_readvariableop_resource+lstm_5_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_5_while_body_328020*$
condR
lstm_5_while_cond_328019*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
lstm_5/while?
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02+
)lstm_5/TensorArrayV2Stack/TensorListStack?
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_5/strided_slice_3/stack?
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_5/strided_slice_3/stack_1?
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_3/stack_2?
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
lstm_5/strided_slice_3?
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose_1/perm?
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
lstm_5/transpose_1t
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/runtime?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMullstm_5/strided_slice_3:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/BiasAdd|
dense_15/SigmoidSigmoiddense_15/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_15/Sigmoid|
dropout_5/IdentityIdentitydense_15/Sigmoid:y:0*
T0*'
_output_shapes
:????????? 2
dropout_5/Identity?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMuldropout_5/Identity:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_16/BiasAdd|
dense_16/SigmoidSigmoiddense_16/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_16/Sigmoid?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_16/Sigmoid:y:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Softmax?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentitydense_17/Softmax:softmax:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp#^lstm_5/lstm_cell_10/ReadVariableOp%^lstm_5/lstm_cell_10/ReadVariableOp_1%^lstm_5/lstm_cell_10/ReadVariableOp_2%^lstm_5/lstm_cell_10/ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp)^lstm_5/lstm_cell_10/split/ReadVariableOp+^lstm_5/lstm_cell_10/split_1/ReadVariableOp^lstm_5/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2H
"lstm_5/lstm_cell_10/ReadVariableOp"lstm_5/lstm_cell_10/ReadVariableOp2L
$lstm_5/lstm_cell_10/ReadVariableOp_1$lstm_5/lstm_cell_10/ReadVariableOp_12L
$lstm_5/lstm_cell_10/ReadVariableOp_2$lstm_5/lstm_cell_10/ReadVariableOp_22L
$lstm_5/lstm_cell_10/ReadVariableOp_3$lstm_5/lstm_cell_10/ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2T
(lstm_5/lstm_cell_10/split/ReadVariableOp(lstm_5/lstm_cell_10/split/ReadVariableOp2X
*lstm_5/lstm_cell_10/split_1/ReadVariableOp*lstm_5/lstm_cell_10/split_1/ReadVariableOp2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
??
?
while_body_328431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_10_split_readvariableop_resource_08
4while_lstm_cell_10_split_1_readvariableop_resource_00
,while_lstm_cell_10_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_10_split_readvariableop_resource6
2while_lstm_cell_10_split_1_readvariableop_resource.
*while_lstm_cell_10_readvariableop_resource??!while/lstm_cell_10/ReadVariableOp?#while/lstm_cell_10/ReadVariableOp_1?#while/lstm_cell_10/ReadVariableOp_2?#while/lstm_cell_10/ReadVariableOp_3?'while/lstm_cell_10/split/ReadVariableOp?)while/lstm_cell_10/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
"while/lstm_cell_10/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/ones_like/Shape?
"while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/ones_like/Const?
while/lstm_cell_10/ones_likeFill+while/lstm_cell_10/ones_like/Shape:output:0+while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/ones_like?
 while/lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/lstm_cell_10/dropout/Const?
while/lstm_cell_10/dropout/MulMul%while/lstm_cell_10/ones_like:output:0)while/lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????&2 
while/lstm_cell_10/dropout/Mul?
 while/lstm_cell_10/dropout/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2"
 while/lstm_cell_10/dropout/Shape?
7while/lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???29
7while/lstm_cell_10/dropout/random_uniform/RandomUniform?
)while/lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)while/lstm_cell_10/dropout/GreaterEqual/y?
'while/lstm_cell_10/dropout/GreaterEqualGreaterEqual@while/lstm_cell_10/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2)
'while/lstm_cell_10/dropout/GreaterEqual?
while/lstm_cell_10/dropout/CastCast+while/lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2!
while/lstm_cell_10/dropout/Cast?
 while/lstm_cell_10/dropout/Mul_1Mul"while/lstm_cell_10/dropout/Mul:z:0#while/lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout/Mul_1?
"while/lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_1/Const?
 while/lstm_cell_10/dropout_1/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout_1/Mul?
"while/lstm_cell_10/dropout_1/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_1/Shape?
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???2;
9while/lstm_cell_10/dropout_1/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_1/GreaterEqual/y?
)while/lstm_cell_10/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2+
)while/lstm_cell_10/dropout_1/GreaterEqual?
!while/lstm_cell_10/dropout_1/CastCast-while/lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2#
!while/lstm_cell_10/dropout_1/Cast?
"while/lstm_cell_10/dropout_1/Mul_1Mul$while/lstm_cell_10/dropout_1/Mul:z:0%while/lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????&2$
"while/lstm_cell_10/dropout_1/Mul_1?
"while/lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_2/Const?
 while/lstm_cell_10/dropout_2/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout_2/Mul?
"while/lstm_cell_10/dropout_2/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_2/Shape?
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2??2;
9while/lstm_cell_10/dropout_2/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_2/GreaterEqual/y?
)while/lstm_cell_10/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2+
)while/lstm_cell_10/dropout_2/GreaterEqual?
!while/lstm_cell_10/dropout_2/CastCast-while/lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2#
!while/lstm_cell_10/dropout_2/Cast?
"while/lstm_cell_10/dropout_2/Mul_1Mul$while/lstm_cell_10/dropout_2/Mul:z:0%while/lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????&2$
"while/lstm_cell_10/dropout_2/Mul_1?
"while/lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_3/Const?
 while/lstm_cell_10/dropout_3/MulMul%while/lstm_cell_10/ones_like:output:0+while/lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????&2"
 while/lstm_cell_10/dropout_3/Mul?
"while/lstm_cell_10/dropout_3/ShapeShape%while/lstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_3/Shape?
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2՞?2;
9while/lstm_cell_10/dropout_3/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_3/GreaterEqual/y?
)while/lstm_cell_10/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2+
)while/lstm_cell_10/dropout_3/GreaterEqual?
!while/lstm_cell_10/dropout_3/CastCast-while/lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2#
!while/lstm_cell_10/dropout_3/Cast?
"while/lstm_cell_10/dropout_3/Mul_1Mul$while/lstm_cell_10/dropout_3/Mul:z:0%while/lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????&2$
"while/lstm_cell_10/dropout_3/Mul_1?
$while/lstm_cell_10/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2&
$while/lstm_cell_10/ones_like_1/Shape?
$while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$while/lstm_cell_10/ones_like_1/Const?
while/lstm_cell_10/ones_like_1Fill-while/lstm_cell_10/ones_like_1/Shape:output:0-while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2 
while/lstm_cell_10/ones_like_1?
"while/lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_4/Const?
 while/lstm_cell_10/dropout_4/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_4/Mul?
"while/lstm_cell_10/dropout_4/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_4/Shape?
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2ބ?2;
9while/lstm_cell_10/dropout_4/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_4/GreaterEqual/y?
)while/lstm_cell_10/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_4/GreaterEqual?
!while/lstm_cell_10/dropout_4/CastCast-while/lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_4/Cast?
"while/lstm_cell_10/dropout_4/Mul_1Mul$while/lstm_cell_10/dropout_4/Mul:z:0%while/lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_4/Mul_1?
"while/lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_5/Const?
 while/lstm_cell_10/dropout_5/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_5/Mul?
"while/lstm_cell_10/dropout_5/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_5/Shape?
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2;
9while/lstm_cell_10/dropout_5/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_5/GreaterEqual/y?
)while/lstm_cell_10/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_5/GreaterEqual?
!while/lstm_cell_10/dropout_5/CastCast-while/lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_5/Cast?
"while/lstm_cell_10/dropout_5/Mul_1Mul$while/lstm_cell_10/dropout_5/Mul:z:0%while/lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_5/Mul_1?
"while/lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_6/Const?
 while/lstm_cell_10/dropout_6/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_6/Mul?
"while/lstm_cell_10/dropout_6/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_6/Shape?
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???2;
9while/lstm_cell_10/dropout_6/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_6/GreaterEqual/y?
)while/lstm_cell_10/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_6/GreaterEqual?
!while/lstm_cell_10/dropout_6/CastCast-while/lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_6/Cast?
"while/lstm_cell_10/dropout_6/Mul_1Mul$while/lstm_cell_10/dropout_6/Mul:z:0%while/lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_6/Mul_1?
"while/lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"while/lstm_cell_10/dropout_7/Const?
 while/lstm_cell_10/dropout_7/MulMul'while/lstm_cell_10/ones_like_1:output:0+while/lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:????????? 2"
 while/lstm_cell_10/dropout_7/Mul?
"while/lstm_cell_10/dropout_7/ShapeShape'while/lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2$
"while/lstm_cell_10/dropout_7/Shape?
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??2;
9while/lstm_cell_10/dropout_7/random_uniform/RandomUniform?
+while/lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+while/lstm_cell_10/dropout_7/GreaterEqual/y?
)while/lstm_cell_10/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2+
)while/lstm_cell_10/dropout_7/GreaterEqual?
!while/lstm_cell_10/dropout_7/CastCast-while/lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2#
!while/lstm_cell_10/dropout_7/Cast?
"while/lstm_cell_10/dropout_7/Mul_1Mul$while/lstm_cell_10/dropout_7/Mul:z:0%while/lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:????????? 2$
"while/lstm_cell_10/dropout_7/Mul_1?
while/lstm_cell_10/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul?
while/lstm_cell_10/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_1?
while/lstm_cell_10/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_2?
while/lstm_cell_10/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
while/lstm_cell_10/mul_3v
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
'while/lstm_cell_10/split/ReadVariableOpReadVariableOp2while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&?*
dtype02)
'while/lstm_cell_10/split/ReadVariableOp?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/MatMulMatMulwhile/lstm_cell_10/mul:z:0!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul?
while/lstm_cell_10/MatMul_1MatMulwhile/lstm_cell_10/mul_1:z:0!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/MatMul_2MatMulwhile/lstm_cell_10/mul_2:z:0!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_2?
while/lstm_cell_10/MatMul_3MatMulwhile/lstm_cell_10/mul_3:z:0!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_3z
while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const_1?
$while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$while/lstm_cell_10/split_1/split_dim?
)while/lstm_cell_10/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/split_1/ReadVariableOp?
while/lstm_cell_10/split_1Split-while/lstm_cell_10/split_1/split_dim:output:01while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
while/lstm_cell_10/split_1?
while/lstm_cell_10/BiasAddBiasAdd#while/lstm_cell_10/MatMul:product:0#while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd?
while/lstm_cell_10/BiasAdd_1BiasAdd%while/lstm_cell_10/MatMul_1:product:0#while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_1?
while/lstm_cell_10/BiasAdd_2BiasAdd%while/lstm_cell_10/MatMul_2:product:0#while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_2?
while/lstm_cell_10/BiasAdd_3BiasAdd%while/lstm_cell_10/MatMul_3:product:0#while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/BiasAdd_3?
while/lstm_cell_10/mul_4Mulwhile_placeholder_2&while/lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_4?
while/lstm_cell_10/mul_5Mulwhile_placeholder_2&while/lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_5?
while/lstm_cell_10/mul_6Mulwhile_placeholder_2&while/lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_6?
while/lstm_cell_10/mul_7Mulwhile_placeholder_2&while/lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_7?
!while/lstm_cell_10/ReadVariableOpReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02#
!while/lstm_cell_10/ReadVariableOp?
&while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&while/lstm_cell_10/strided_slice/stack?
(while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice/stack_1?
(while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(while/lstm_cell_10/strided_slice/stack_2?
 while/lstm_cell_10/strided_sliceStridedSlice)while/lstm_cell_10/ReadVariableOp:value:0/while/lstm_cell_10/strided_slice/stack:output:01while/lstm_cell_10/strided_slice/stack_1:output:01while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2"
 while/lstm_cell_10/strided_slice?
while/lstm_cell_10/MatMul_4MatMulwhile/lstm_cell_10/mul_4:z:0)while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_4?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/BiasAdd:output:0%while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add?
while/lstm_cell_10/SigmoidSigmoidwhile/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid?
#while/lstm_cell_10/ReadVariableOp_1ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_1?
(while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(while/lstm_cell_10/strided_slice_1/stack?
*while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2,
*while/lstm_cell_10/strided_slice_1/stack_1?
*while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_1/stack_2?
"while/lstm_cell_10/strided_slice_1StridedSlice+while/lstm_cell_10/ReadVariableOp_1:value:01while/lstm_cell_10/strided_slice_1/stack:output:03while/lstm_cell_10/strided_slice_1/stack_1:output:03while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_1?
while/lstm_cell_10/MatMul_5MatMulwhile/lstm_cell_10/mul_5:z:0+while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_5?
while/lstm_cell_10/add_1AddV2%while/lstm_cell_10/BiasAdd_1:output:0%while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_1Sigmoidwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mul_8Mul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_8?
#while/lstm_cell_10/ReadVariableOp_2ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_2?
(while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(while/lstm_cell_10/strided_slice_2/stack?
*while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2,
*while/lstm_cell_10/strided_slice_2/stack_1?
*while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_2/stack_2?
"while/lstm_cell_10/strided_slice_2StridedSlice+while/lstm_cell_10/ReadVariableOp_2:value:01while/lstm_cell_10/strided_slice_2/stack:output:03while/lstm_cell_10/strided_slice_2/stack_1:output:03while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_2?
while/lstm_cell_10/MatMul_6MatMulwhile/lstm_cell_10/mul_6:z:0+while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_6?
while/lstm_cell_10/add_2AddV2%while/lstm_cell_10/BiasAdd_2:output:0%while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_2?
while/lstm_cell_10/TanhTanhwhile/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh?
while/lstm_cell_10/mul_9Mulwhile/lstm_cell_10/Sigmoid:y:0while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_9?
while/lstm_cell_10/add_3AddV2while/lstm_cell_10/mul_8:z:0while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_3?
#while/lstm_cell_10/ReadVariableOp_3ReadVariableOp,while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02%
#while/lstm_cell_10/ReadVariableOp_3?
(while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2*
(while/lstm_cell_10/strided_slice_3/stack?
*while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*while/lstm_cell_10/strided_slice_3/stack_1?
*while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*while/lstm_cell_10/strided_slice_3/stack_2?
"while/lstm_cell_10/strided_slice_3StridedSlice+while/lstm_cell_10/ReadVariableOp_3:value:01while/lstm_cell_10/strided_slice_3/stack:output:03while/lstm_cell_10/strided_slice_3/stack_1:output:03while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2$
"while/lstm_cell_10/strided_slice_3?
while/lstm_cell_10/MatMul_7MatMulwhile/lstm_cell_10/mul_7:z:0+while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/MatMul_7?
while/lstm_cell_10/add_4AddV2%while/lstm_cell_10/BiasAdd_3:output:0%while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/add_4?
while/lstm_cell_10/Sigmoid_2Sigmoidwhile/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Tanh_1Tanhwhile/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/Tanh_1?
while/lstm_cell_10/mul_10Mul while/lstm_cell_10/Sigmoid_2:y:0while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_10/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_10:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_3:z:0"^while/lstm_cell_10/ReadVariableOp$^while/lstm_cell_10/ReadVariableOp_1$^while/lstm_cell_10/ReadVariableOp_2$^while/lstm_cell_10/ReadVariableOp_3(^while/lstm_cell_10/split/ReadVariableOp*^while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_10_readvariableop_resource,while_lstm_cell_10_readvariableop_resource_0"j
2while_lstm_cell_10_split_1_readvariableop_resource4while_lstm_cell_10_split_1_readvariableop_resource_0"f
0while_lstm_cell_10_split_readvariableop_resource2while_lstm_cell_10_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2F
!while/lstm_cell_10/ReadVariableOp!while/lstm_cell_10/ReadVariableOp2J
#while/lstm_cell_10/ReadVariableOp_1#while/lstm_cell_10/ReadVariableOp_12J
#while/lstm_cell_10/ReadVariableOp_2#while/lstm_cell_10/ReadVariableOp_22J
#while/lstm_cell_10/ReadVariableOp_3#while/lstm_cell_10/ReadVariableOp_32R
'while/lstm_cell_10/split/ReadVariableOp'while/lstm_cell_10/split/ReadVariableOp2V
)while/lstm_cell_10/split_1/ReadVariableOp)while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_329648

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_5_layer_call_fn_327370
lstm_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_3273492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????&
&
_user_specified_namelstm_5_input
??
?	
!__inference__wrapped_model_325601
lstm_5_inputB
>sequential_5_lstm_5_lstm_cell_10_split_readvariableop_resourceD
@sequential_5_lstm_5_lstm_cell_10_split_1_readvariableop_resource<
8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource8
4sequential_5_dense_15_matmul_readvariableop_resource9
5sequential_5_dense_15_biasadd_readvariableop_resource8
4sequential_5_dense_16_matmul_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource8
4sequential_5_dense_17_matmul_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource
identity??,sequential_5/dense_15/BiasAdd/ReadVariableOp?+sequential_5/dense_15/MatMul/ReadVariableOp?,sequential_5/dense_16/BiasAdd/ReadVariableOp?+sequential_5/dense_16/MatMul/ReadVariableOp?,sequential_5/dense_17/BiasAdd/ReadVariableOp?+sequential_5/dense_17/MatMul/ReadVariableOp?/sequential_5/lstm_5/lstm_cell_10/ReadVariableOp?1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_1?1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_2?1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_3?5sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOp?7sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOp?sequential_5/lstm_5/whiler
sequential_5/lstm_5/ShapeShapelstm_5_input*
T0*
_output_shapes
:2
sequential_5/lstm_5/Shape?
'sequential_5/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/lstm_5/strided_slice/stack?
)sequential_5/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_5/strided_slice/stack_1?
)sequential_5/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_5/strided_slice/stack_2?
!sequential_5/lstm_5/strided_sliceStridedSlice"sequential_5/lstm_5/Shape:output:00sequential_5/lstm_5/strided_slice/stack:output:02sequential_5/lstm_5/strided_slice/stack_1:output:02sequential_5/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_5/lstm_5/strided_slice?
sequential_5/lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_5/lstm_5/zeros/mul/y?
sequential_5/lstm_5/zeros/mulMul*sequential_5/lstm_5/strided_slice:output:0(sequential_5/lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_5/zeros/mul?
 sequential_5/lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_5/lstm_5/zeros/Less/y?
sequential_5/lstm_5/zeros/LessLess!sequential_5/lstm_5/zeros/mul:z:0)sequential_5/lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_5/zeros/Less?
"sequential_5/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_5/lstm_5/zeros/packed/1?
 sequential_5/lstm_5/zeros/packedPack*sequential_5/lstm_5/strided_slice:output:0+sequential_5/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_5/lstm_5/zeros/packed?
sequential_5/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_5/lstm_5/zeros/Const?
sequential_5/lstm_5/zerosFill)sequential_5/lstm_5/zeros/packed:output:0(sequential_5/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
sequential_5/lstm_5/zeros?
!sequential_5/lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_5/lstm_5/zeros_1/mul/y?
sequential_5/lstm_5/zeros_1/mulMul*sequential_5/lstm_5/strided_slice:output:0*sequential_5/lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_5/zeros_1/mul?
"sequential_5/lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_5/lstm_5/zeros_1/Less/y?
 sequential_5/lstm_5/zeros_1/LessLess#sequential_5/lstm_5/zeros_1/mul:z:0+sequential_5/lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_5/zeros_1/Less?
$sequential_5/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_5/lstm_5/zeros_1/packed/1?
"sequential_5/lstm_5/zeros_1/packedPack*sequential_5/lstm_5/strided_slice:output:0-sequential_5/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_5/lstm_5/zeros_1/packed?
!sequential_5/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_5/lstm_5/zeros_1/Const?
sequential_5/lstm_5/zeros_1Fill+sequential_5/lstm_5/zeros_1/packed:output:0*sequential_5/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
sequential_5/lstm_5/zeros_1?
"sequential_5/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_5/lstm_5/transpose/perm?
sequential_5/lstm_5/transpose	Transposelstm_5_input+sequential_5/lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????&2
sequential_5/lstm_5/transpose?
sequential_5/lstm_5/Shape_1Shape!sequential_5/lstm_5/transpose:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_5/Shape_1?
)sequential_5/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_5/strided_slice_1/stack?
+sequential_5/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_1/stack_1?
+sequential_5/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_1/stack_2?
#sequential_5/lstm_5/strided_slice_1StridedSlice$sequential_5/lstm_5/Shape_1:output:02sequential_5/lstm_5/strided_slice_1/stack:output:04sequential_5/lstm_5/strided_slice_1/stack_1:output:04sequential_5/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_5/lstm_5/strided_slice_1?
/sequential_5/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_5/lstm_5/TensorArrayV2/element_shape?
!sequential_5/lstm_5/TensorArrayV2TensorListReserve8sequential_5/lstm_5/TensorArrayV2/element_shape:output:0,sequential_5/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_5/lstm_5/TensorArrayV2?
Isequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   2K
Isequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/lstm_5/transpose:y:0Rsequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor?
)sequential_5/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_5/strided_slice_2/stack?
+sequential_5/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_2/stack_1?
+sequential_5/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_2/stack_2?
#sequential_5/lstm_5/strided_slice_2StridedSlice!sequential_5/lstm_5/transpose:y:02sequential_5/lstm_5/strided_slice_2/stack:output:04sequential_5/lstm_5/strided_slice_2/stack_1:output:04sequential_5/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2%
#sequential_5/lstm_5/strided_slice_2?
0sequential_5/lstm_5/lstm_cell_10/ones_like/ShapeShape,sequential_5/lstm_5/strided_slice_2:output:0*
T0*
_output_shapes
:22
0sequential_5/lstm_5/lstm_cell_10/ones_like/Shape?
0sequential_5/lstm_5/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential_5/lstm_5/lstm_cell_10/ones_like/Const?
*sequential_5/lstm_5/lstm_cell_10/ones_likeFill9sequential_5/lstm_5/lstm_cell_10/ones_like/Shape:output:09sequential_5/lstm_5/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2,
*sequential_5/lstm_5/lstm_cell_10/ones_like?
2sequential_5/lstm_5/lstm_cell_10/ones_like_1/ShapeShape"sequential_5/lstm_5/zeros:output:0*
T0*
_output_shapes
:24
2sequential_5/lstm_5/lstm_cell_10/ones_like_1/Shape?
2sequential_5/lstm_5/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_5/lstm_5/lstm_cell_10/ones_like_1/Const?
,sequential_5/lstm_5/lstm_cell_10/ones_like_1Fill;sequential_5/lstm_5/lstm_cell_10/ones_like_1/Shape:output:0;sequential_5/lstm_5/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/lstm_cell_10/ones_like_1?
$sequential_5/lstm_5/lstm_cell_10/mulMul,sequential_5/lstm_5/strided_slice_2:output:03sequential_5/lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2&
$sequential_5/lstm_5/lstm_cell_10/mul?
&sequential_5/lstm_5/lstm_cell_10/mul_1Mul,sequential_5/lstm_5/strided_slice_2:output:03sequential_5/lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2(
&sequential_5/lstm_5/lstm_cell_10/mul_1?
&sequential_5/lstm_5/lstm_cell_10/mul_2Mul,sequential_5/lstm_5/strided_slice_2:output:03sequential_5/lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2(
&sequential_5/lstm_5/lstm_cell_10/mul_2?
&sequential_5/lstm_5/lstm_cell_10/mul_3Mul,sequential_5/lstm_5/strided_slice_2:output:03sequential_5/lstm_5/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2(
&sequential_5/lstm_5/lstm_cell_10/mul_3?
&sequential_5/lstm_5/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_5/lstm_5/lstm_cell_10/Const?
0sequential_5/lstm_5/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential_5/lstm_5/lstm_cell_10/split/split_dim?
5sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOpReadVariableOp>sequential_5_lstm_5_lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype027
5sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOp?
&sequential_5/lstm_5/lstm_cell_10/splitSplit9sequential_5/lstm_5/lstm_cell_10/split/split_dim:output:0=sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2(
&sequential_5/lstm_5/lstm_cell_10/split?
'sequential_5/lstm_5/lstm_cell_10/MatMulMatMul(sequential_5/lstm_5/lstm_cell_10/mul:z:0/sequential_5/lstm_5/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2)
'sequential_5/lstm_5/lstm_cell_10/MatMul?
)sequential_5/lstm_5/lstm_cell_10/MatMul_1MatMul*sequential_5/lstm_5/lstm_cell_10/mul_1:z:0/sequential_5/lstm_5/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_1?
)sequential_5/lstm_5/lstm_cell_10/MatMul_2MatMul*sequential_5/lstm_5/lstm_cell_10/mul_2:z:0/sequential_5/lstm_5/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_2?
)sequential_5/lstm_5/lstm_cell_10/MatMul_3MatMul*sequential_5/lstm_5/lstm_cell_10/mul_3:z:0/sequential_5/lstm_5/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_3?
(sequential_5/lstm_5/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_5/lstm_5/lstm_cell_10/Const_1?
2sequential_5/lstm_5/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2sequential_5/lstm_5/lstm_cell_10/split_1/split_dim?
7sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOpReadVariableOp@sequential_5_lstm_5_lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOp?
(sequential_5/lstm_5/lstm_cell_10/split_1Split;sequential_5/lstm_5/lstm_cell_10/split_1/split_dim:output:0?sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2*
(sequential_5/lstm_5/lstm_cell_10/split_1?
(sequential_5/lstm_5/lstm_cell_10/BiasAddBiasAdd1sequential_5/lstm_5/lstm_cell_10/MatMul:product:01sequential_5/lstm_5/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2*
(sequential_5/lstm_5/lstm_cell_10/BiasAdd?
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_1BiasAdd3sequential_5/lstm_5/lstm_cell_10/MatMul_1:product:01sequential_5/lstm_5/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2,
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_1?
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_2BiasAdd3sequential_5/lstm_5/lstm_cell_10/MatMul_2:product:01sequential_5/lstm_5/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2,
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_2?
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_3BiasAdd3sequential_5/lstm_5/lstm_cell_10/MatMul_3:product:01sequential_5/lstm_5/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2,
*sequential_5/lstm_5/lstm_cell_10/BiasAdd_3?
&sequential_5/lstm_5/lstm_cell_10/mul_4Mul"sequential_5/lstm_5/zeros:output:05sequential_5/lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2(
&sequential_5/lstm_5/lstm_cell_10/mul_4?
&sequential_5/lstm_5/lstm_cell_10/mul_5Mul"sequential_5/lstm_5/zeros:output:05sequential_5/lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2(
&sequential_5/lstm_5/lstm_cell_10/mul_5?
&sequential_5/lstm_5/lstm_cell_10/mul_6Mul"sequential_5/lstm_5/zeros:output:05sequential_5/lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2(
&sequential_5/lstm_5/lstm_cell_10/mul_6?
&sequential_5/lstm_5/lstm_cell_10/mul_7Mul"sequential_5/lstm_5/zeros:output:05sequential_5/lstm_5/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2(
&sequential_5/lstm_5/lstm_cell_10/mul_7?
/sequential_5/lstm_5/lstm_cell_10/ReadVariableOpReadVariableOp8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype021
/sequential_5/lstm_5/lstm_cell_10/ReadVariableOp?
4sequential_5/lstm_5/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4sequential_5/lstm_5/lstm_cell_10/strided_slice/stack?
6sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        28
6sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_1?
6sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_2?
.sequential_5/lstm_5/lstm_cell_10/strided_sliceStridedSlice7sequential_5/lstm_5/lstm_cell_10/ReadVariableOp:value:0=sequential_5/lstm_5/lstm_cell_10/strided_slice/stack:output:0?sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_1:output:0?sequential_5/lstm_5/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask20
.sequential_5/lstm_5/lstm_cell_10/strided_slice?
)sequential_5/lstm_5/lstm_cell_10/MatMul_4MatMul*sequential_5/lstm_5/lstm_cell_10/mul_4:z:07sequential_5/lstm_5/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_4?
$sequential_5/lstm_5/lstm_cell_10/addAddV21sequential_5/lstm_5/lstm_cell_10/BiasAdd:output:03sequential_5/lstm_5/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2&
$sequential_5/lstm_5/lstm_cell_10/add?
(sequential_5/lstm_5/lstm_cell_10/SigmoidSigmoid(sequential_5/lstm_5/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2*
(sequential_5/lstm_5/lstm_cell_10/Sigmoid?
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_1ReadVariableOp8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype023
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_1?
6sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        28
6sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack?
8sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_1?
8sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_2?
0sequential_5/lstm_5/lstm_cell_10/strided_slice_1StridedSlice9sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_1:value:0?sequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_1:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask22
0sequential_5/lstm_5/lstm_cell_10/strided_slice_1?
)sequential_5/lstm_5/lstm_cell_10/MatMul_5MatMul*sequential_5/lstm_5/lstm_cell_10/mul_5:z:09sequential_5/lstm_5/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_5?
&sequential_5/lstm_5/lstm_cell_10/add_1AddV23sequential_5/lstm_5/lstm_cell_10/BiasAdd_1:output:03sequential_5/lstm_5/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2(
&sequential_5/lstm_5/lstm_cell_10/add_1?
*sequential_5/lstm_5/lstm_cell_10/Sigmoid_1Sigmoid*sequential_5/lstm_5/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2,
*sequential_5/lstm_5/lstm_cell_10/Sigmoid_1?
&sequential_5/lstm_5/lstm_cell_10/mul_8Mul.sequential_5/lstm_5/lstm_cell_10/Sigmoid_1:y:0$sequential_5/lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2(
&sequential_5/lstm_5/lstm_cell_10/mul_8?
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_2ReadVariableOp8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype023
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_2?
6sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   28
6sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack?
8sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_1?
8sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_2?
0sequential_5/lstm_5/lstm_cell_10/strided_slice_2StridedSlice9sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_2:value:0?sequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_1:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask22
0sequential_5/lstm_5/lstm_cell_10/strided_slice_2?
)sequential_5/lstm_5/lstm_cell_10/MatMul_6MatMul*sequential_5/lstm_5/lstm_cell_10/mul_6:z:09sequential_5/lstm_5/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_6?
&sequential_5/lstm_5/lstm_cell_10/add_2AddV23sequential_5/lstm_5/lstm_cell_10/BiasAdd_2:output:03sequential_5/lstm_5/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2(
&sequential_5/lstm_5/lstm_cell_10/add_2?
%sequential_5/lstm_5/lstm_cell_10/TanhTanh*sequential_5/lstm_5/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2'
%sequential_5/lstm_5/lstm_cell_10/Tanh?
&sequential_5/lstm_5/lstm_cell_10/mul_9Mul,sequential_5/lstm_5/lstm_cell_10/Sigmoid:y:0)sequential_5/lstm_5/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2(
&sequential_5/lstm_5/lstm_cell_10/mul_9?
&sequential_5/lstm_5/lstm_cell_10/add_3AddV2*sequential_5/lstm_5/lstm_cell_10/mul_8:z:0*sequential_5/lstm_5/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2(
&sequential_5/lstm_5/lstm_cell_10/add_3?
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_3ReadVariableOp8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype023
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_3?
6sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   28
6sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack?
8sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_1?
8sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_2?
0sequential_5/lstm_5/lstm_cell_10/strided_slice_3StridedSlice9sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_3:value:0?sequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_1:output:0Asequential_5/lstm_5/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask22
0sequential_5/lstm_5/lstm_cell_10/strided_slice_3?
)sequential_5/lstm_5/lstm_cell_10/MatMul_7MatMul*sequential_5/lstm_5/lstm_cell_10/mul_7:z:09sequential_5/lstm_5/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2+
)sequential_5/lstm_5/lstm_cell_10/MatMul_7?
&sequential_5/lstm_5/lstm_cell_10/add_4AddV23sequential_5/lstm_5/lstm_cell_10/BiasAdd_3:output:03sequential_5/lstm_5/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2(
&sequential_5/lstm_5/lstm_cell_10/add_4?
*sequential_5/lstm_5/lstm_cell_10/Sigmoid_2Sigmoid*sequential_5/lstm_5/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2,
*sequential_5/lstm_5/lstm_cell_10/Sigmoid_2?
'sequential_5/lstm_5/lstm_cell_10/Tanh_1Tanh*sequential_5/lstm_5/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2)
'sequential_5/lstm_5/lstm_cell_10/Tanh_1?
'sequential_5/lstm_5/lstm_cell_10/mul_10Mul.sequential_5/lstm_5/lstm_cell_10/Sigmoid_2:y:0+sequential_5/lstm_5/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2)
'sequential_5/lstm_5/lstm_cell_10/mul_10?
1sequential_5/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    23
1sequential_5/lstm_5/TensorArrayV2_1/element_shape?
#sequential_5/lstm_5/TensorArrayV2_1TensorListReserve:sequential_5/lstm_5/TensorArrayV2_1/element_shape:output:0,sequential_5/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_5/lstm_5/TensorArrayV2_1v
sequential_5/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_5/lstm_5/time?
,sequential_5/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_5/lstm_5/while/maximum_iterations?
&sequential_5/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_5/lstm_5/while/loop_counter?
sequential_5/lstm_5/whileWhile/sequential_5/lstm_5/while/loop_counter:output:05sequential_5/lstm_5/while/maximum_iterations:output:0!sequential_5/lstm_5/time:output:0,sequential_5/lstm_5/TensorArrayV2_1:handle:0"sequential_5/lstm_5/zeros:output:0$sequential_5/lstm_5/zeros_1:output:0,sequential_5/lstm_5/strided_slice_1:output:0Ksequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_5_lstm_5_lstm_cell_10_split_readvariableop_resource@sequential_5_lstm_5_lstm_cell_10_split_1_readvariableop_resource8sequential_5_lstm_5_lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*1
body)R'
%sequential_5_lstm_5_while_body_325443*1
cond)R'
%sequential_5_lstm_5_while_cond_325442*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
sequential_5/lstm_5/while?
Dsequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2F
Dsequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_5/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/lstm_5/while:output:3Msequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype028
6sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack?
)sequential_5/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_5/lstm_5/strided_slice_3/stack?
+sequential_5/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_5/lstm_5/strided_slice_3/stack_1?
+sequential_5/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_3/stack_2?
#sequential_5/lstm_5/strided_slice_3StridedSlice?sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/lstm_5/strided_slice_3/stack:output:04sequential_5/lstm_5/strided_slice_3/stack_1:output:04sequential_5/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2%
#sequential_5/lstm_5/strided_slice_3?
$sequential_5/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_5/lstm_5/transpose_1/perm?
sequential_5/lstm_5/transpose_1	Transpose?sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2!
sequential_5/lstm_5/transpose_1?
sequential_5/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_5/lstm_5/runtime?
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOp?
sequential_5/dense_15/MatMulMatMul,sequential_5/lstm_5/strided_slice_3:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_15/MatMul?
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOp?
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_15/BiasAdd?
sequential_5/dense_15/SigmoidSigmoid&sequential_5/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_15/Sigmoid?
sequential_5/dropout_5/IdentityIdentity!sequential_5/dense_15/Sigmoid:y:0*
T0*'
_output_shapes
:????????? 2!
sequential_5/dropout_5/Identity?
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOp?
sequential_5/dense_16/MatMulMatMul(sequential_5/dropout_5/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_16/MatMul?
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp?
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_16/BiasAdd?
sequential_5/dense_16/SigmoidSigmoid&sequential_5/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_16/Sigmoid?
+sequential_5/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_17_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_5/dense_17/MatMul/ReadVariableOp?
sequential_5/dense_17/MatMulMatMul!sequential_5/dense_16/Sigmoid:y:03sequential_5/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_17/MatMul?
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOp?
sequential_5/dense_17/BiasAddBiasAdd&sequential_5/dense_17/MatMul:product:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_17/BiasAdd?
sequential_5/dense_17/SoftmaxSoftmax&sequential_5/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_17/Softmax?
IdentityIdentity'sequential_5/dense_17/Softmax:softmax:0-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp,^sequential_5/dense_17/MatMul/ReadVariableOp0^sequential_5/lstm_5/lstm_cell_10/ReadVariableOp2^sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_12^sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_22^sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_36^sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOp8^sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOp^sequential_5/lstm_5/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_17/MatMul/ReadVariableOp+sequential_5/dense_17/MatMul/ReadVariableOp2b
/sequential_5/lstm_5/lstm_cell_10/ReadVariableOp/sequential_5/lstm_5/lstm_cell_10/ReadVariableOp2f
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_11sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_12f
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_21sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_22f
1sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_31sequential_5/lstm_5/lstm_cell_10/ReadVariableOp_32n
5sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOp5sequential_5/lstm_5/lstm_cell_10/split/ReadVariableOp2r
7sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOp7sequential_5/lstm_5/lstm_cell_10/split_1/ReadVariableOp26
sequential_5/lstm_5/whilesequential_5/lstm_5/while:Y U
+
_output_shapes
:?????????&
&
_user_specified_namelstm_5_input
?
?
while_cond_328430
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_328430___redundant_placeholder04
0while_while_cond_328430___redundant_placeholder14
0while_while_cond_328430___redundant_placeholder24
0while_while_cond_328430___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
޵
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_329594

inputs.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identity??:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?lstm_cell_10/ReadVariableOp?lstm_cell_10/ReadVariableOp_1?lstm_cell_10/ReadVariableOp_2?lstm_cell_10/ReadVariableOp_3?!lstm_cell_10/split/ReadVariableOp?#lstm_cell_10/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2
strided_slice_2?
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape?
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/ones_like/Const?
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/ones_like~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape?
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm_cell_10/ones_like_1/Const?
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/ones_like_1?
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul?
lstm_cell_10/mul_1Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_1?
lstm_cell_10/mul_2Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_2?
lstm_cell_10/mul_3Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02#
!lstm_cell_10/split/ReadVariableOp?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split?
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul?
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_1?
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_2?
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1?
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dim?
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp?
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd?
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_1?
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_2?
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_3?
lstm_cell_10/mul_4Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_4?
lstm_cell_10/mul_5Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_5?
lstm_cell_10/mul_6Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_6?
lstm_cell_10/mul_7Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_7?
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp?
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack?
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1?
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2?
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice?
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_4?
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid?
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_1?
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack?
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1?
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2?
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1?
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_5?
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_8?
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_2?
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack?
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1?
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2?
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2?
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_6?
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh?
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_9?
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_3?
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_3?
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack?
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1?
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2?
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3?
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_7?
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_4?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh_1?
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_329446*
condR
while_cond_329445*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
?	
?
lstm_5_while_cond_327659*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1B
>lstm_5_while_lstm_5_while_cond_327659___redundant_placeholder0B
>lstm_5_while_lstm_5_while_cond_327659___redundant_placeholder1B
>lstm_5_while_lstm_5_while_cond_327659___redundant_placeholder2B
>lstm_5_while_lstm_5_while_cond_327659___redundant_placeholder3
lstm_5_while_identity
?
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
lstm_5/while/Lessr
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_5/while/Identity"7
lstm_5_while_identitylstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?	
?
D__inference_dense_15_layer_call_and_return_conditional_losses_327155

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_lstm_cell_10_layer_call_fn_329988

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_3258012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????&:????????? :????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?
?
while_cond_326634
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_326634___redundant_placeholder04
0while_while_cond_326634___redundant_placeholder14
0while_while_cond_326634___redundant_placeholder24
0while_while_cond_326634___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_327183

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?L
?
__inference__traced_save_330158
file_prefix.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop#
savev2_iter_read_readvariableop	%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop9
5savev2_lstm_5_lstm_cell_10_kernel_read_readvariableopC
?savev2_lstm_5_lstm_cell_10_recurrent_kernel_read_readvariableop7
3savev2_lstm_5_lstm_cell_10_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop0
,savev2_dense_15_kernel_m_read_readvariableop.
*savev2_dense_15_bias_m_read_readvariableop0
,savev2_dense_16_kernel_m_read_readvariableop.
*savev2_dense_16_bias_m_read_readvariableop0
,savev2_dense_17_kernel_m_read_readvariableop.
*savev2_dense_17_bias_m_read_readvariableop;
7savev2_lstm_5_lstm_cell_10_kernel_m_read_readvariableopE
Asavev2_lstm_5_lstm_cell_10_recurrent_kernel_m_read_readvariableop9
5savev2_lstm_5_lstm_cell_10_bias_m_read_readvariableop0
,savev2_dense_15_kernel_v_read_readvariableop.
*savev2_dense_15_bias_v_read_readvariableop0
,savev2_dense_16_kernel_v_read_readvariableop.
*savev2_dense_16_bias_v_read_readvariableop0
,savev2_dense_17_kernel_v_read_readvariableop.
*savev2_dense_17_bias_v_read_readvariableop;
7savev2_lstm_5_lstm_cell_10_kernel_v_read_readvariableopE
Asavev2_lstm_5_lstm_cell_10_recurrent_kernel_v_read_readvariableop9
5savev2_lstm_5_lstm_cell_10_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*?
value?B?%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableopsavev2_iter_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop5savev2_lstm_5_lstm_cell_10_kernel_read_readvariableop?savev2_lstm_5_lstm_cell_10_recurrent_kernel_read_readvariableop3savev2_lstm_5_lstm_cell_10_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop,savev2_dense_15_kernel_m_read_readvariableop*savev2_dense_15_bias_m_read_readvariableop,savev2_dense_16_kernel_m_read_readvariableop*savev2_dense_16_bias_m_read_readvariableop,savev2_dense_17_kernel_m_read_readvariableop*savev2_dense_17_bias_m_read_readvariableop7savev2_lstm_5_lstm_cell_10_kernel_m_read_readvariableopAsavev2_lstm_5_lstm_cell_10_recurrent_kernel_m_read_readvariableop5savev2_lstm_5_lstm_cell_10_bias_m_read_readvariableop,savev2_dense_15_kernel_v_read_readvariableop*savev2_dense_15_bias_v_read_readvariableop,savev2_dense_16_kernel_v_read_readvariableop*savev2_dense_16_bias_v_read_readvariableop,savev2_dense_17_kernel_v_read_readvariableop*savev2_dense_17_bias_v_read_readvariableop7savev2_lstm_5_lstm_cell_10_kernel_v_read_readvariableopAsavev2_lstm_5_lstm_cell_10_recurrent_kernel_v_read_readvariableop5savev2_lstm_5_lstm_cell_10_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :  : :  : : :: : : : : :	&?:	 ?:?: : : : :  : :  : : ::	&?:	 ?:?:  : :  : : ::	&?:	 ?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	&?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	&?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$  

_output_shapes

: : !

_output_shapes
::%"!

_output_shapes
:	&?:%#!

_output_shapes
:	 ?:!$

_output_shapes	
:?:%

_output_shapes
: 
?
?
'__inference_lstm_5_layer_call_fn_329616

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_3271142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_330016I
Elstm_5_lstm_cell_10_kernel_regularizer_square_readvariableop_resource
identity??<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpElstm_5_lstm_cell_10_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
IdentityIdentity.lstm_5/lstm_cell_10/kernel/Regularizer/mul:z:0=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp
?	
?
D__inference_dense_15_layer_call_and_return_conditional_losses_329627

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?%
?
while_body_326215
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_10_326239_0
while_lstm_cell_10_326241_0
while_lstm_cell_10_326243_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_10_326239
while_lstm_cell_10_326241
while_lstm_cell_10_326243??*while/lstm_cell_10/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_326239_0while_lstm_cell_10_326241_0while_lstm_cell_10_326243_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_3258012,
*while/lstm_cell_10/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_10_326239while_lstm_cell_10_326239_0"8
while_lstm_cell_10_326241while_lstm_cell_10_326241_0"8
while_lstm_cell_10_326243while_lstm_cell_10_326243_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
~
)__inference_dense_17_layer_call_fn_329703

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_3272392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?0
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_327411

inputs
lstm_5_327375
lstm_5_327377
lstm_5_327379
dense_15_327382
dense_15_327384
dense_16_327388
dense_16_327390
dense_17_327393
dense_17_327395
identity?? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
lstm_5/StatefulPartitionedCallStatefulPartitionedCallinputslstm_5_327375lstm_5_327377lstm_5_327379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_3271142 
lstm_5/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_15_327382dense_15_327384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_3271552"
 dense_15/StatefulPartitionedCall?
dropout_5/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3271882
dropout_5/PartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_16_327388dense_16_327390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_3272122"
 dense_16/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_327393dense_17_327395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_3272392"
 dense_17/StatefulPartitionedCall?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_327375*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_5_327377*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
-__inference_sequential_5_layer_call_fn_328213

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_3273492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????&:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????&
 
_user_specified_nameinputs
Ғ
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_328643
inputs_0.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identity??:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?lstm_cell_10/ReadVariableOp?lstm_cell_10/ReadVariableOp_1?lstm_cell_10/ReadVariableOp_2?lstm_cell_10/ReadVariableOp_3?!lstm_cell_10/split/ReadVariableOp?#lstm_cell_10/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2
strided_slice_2?
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape?
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/ones_like/Const?
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/ones_like}
lstm_cell_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout/Const?
lstm_cell_10/dropout/MulMullstm_cell_10/ones_like:output:0#lstm_cell_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout/Mul?
lstm_cell_10/dropout/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout/Shape?
1lstm_cell_10/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2??c23
1lstm_cell_10/dropout/random_uniform/RandomUniform?
#lstm_cell_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2%
#lstm_cell_10/dropout/GreaterEqual/y?
!lstm_cell_10/dropout/GreaterEqualGreaterEqual:lstm_cell_10/dropout/random_uniform/RandomUniform:output:0,lstm_cell_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2#
!lstm_cell_10/dropout/GreaterEqual?
lstm_cell_10/dropout/CastCast%lstm_cell_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout/Cast?
lstm_cell_10/dropout/Mul_1Mullstm_cell_10/dropout/Mul:z:0lstm_cell_10/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout/Mul_1?
lstm_cell_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_1/Const?
lstm_cell_10/dropout_1/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_1/Mul?
lstm_cell_10/dropout_1/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_1/Shape?
3lstm_cell_10/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_1/random_uniform/RandomUniform?
%lstm_cell_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_1/GreaterEqual/y?
#lstm_cell_10/dropout_1/GreaterEqualGreaterEqual<lstm_cell_10/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_cell_10/dropout_1/GreaterEqual?
lstm_cell_10/dropout_1/CastCast'lstm_cell_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_1/Cast?
lstm_cell_10/dropout_1/Mul_1Mullstm_cell_10/dropout_1/Mul:z:0lstm_cell_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_1/Mul_1?
lstm_cell_10/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_2/Const?
lstm_cell_10/dropout_2/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_2/Mul?
lstm_cell_10/dropout_2/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_2/Shape?
3lstm_cell_10/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2?Ħ25
3lstm_cell_10/dropout_2/random_uniform/RandomUniform?
%lstm_cell_10/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_2/GreaterEqual/y?
#lstm_cell_10/dropout_2/GreaterEqualGreaterEqual<lstm_cell_10/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_cell_10/dropout_2/GreaterEqual?
lstm_cell_10/dropout_2/CastCast'lstm_cell_10/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_2/Cast?
lstm_cell_10/dropout_2/Mul_1Mullstm_cell_10/dropout_2/Mul:z:0lstm_cell_10/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_2/Mul_1?
lstm_cell_10/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_3/Const?
lstm_cell_10/dropout_3/MulMullstm_cell_10/ones_like:output:0%lstm_cell_10/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_3/Mul?
lstm_cell_10/dropout_3/ShapeShapelstm_cell_10/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_3/Shape?
3lstm_cell_10/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????&*
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_3/random_uniform/RandomUniform?
%lstm_cell_10/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_3/GreaterEqual/y?
#lstm_cell_10/dropout_3/GreaterEqualGreaterEqual<lstm_cell_10/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????&2%
#lstm_cell_10/dropout_3/GreaterEqual?
lstm_cell_10/dropout_3/CastCast'lstm_cell_10/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_3/Cast?
lstm_cell_10/dropout_3/Mul_1Mullstm_cell_10/dropout_3/Mul:z:0lstm_cell_10/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/dropout_3/Mul_1~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape?
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm_cell_10/ones_like_1/Const?
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/ones_like_1?
lstm_cell_10/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_4/Const?
lstm_cell_10/dropout_4/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_4/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_4/Mul?
lstm_cell_10/dropout_4/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_4/Shape?
3lstm_cell_10/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_4/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_4/random_uniform/RandomUniform?
%lstm_cell_10/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_4/GreaterEqual/y?
#lstm_cell_10/dropout_4/GreaterEqualGreaterEqual<lstm_cell_10/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_4/GreaterEqual?
lstm_cell_10/dropout_4/CastCast'lstm_cell_10/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_4/Cast?
lstm_cell_10/dropout_4/Mul_1Mullstm_cell_10/dropout_4/Mul:z:0lstm_cell_10/dropout_4/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_4/Mul_1?
lstm_cell_10/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_5/Const?
lstm_cell_10/dropout_5/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_5/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_5/Mul?
lstm_cell_10/dropout_5/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_5/Shape?
3lstm_cell_10/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_5/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_5/random_uniform/RandomUniform?
%lstm_cell_10/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_5/GreaterEqual/y?
#lstm_cell_10/dropout_5/GreaterEqualGreaterEqual<lstm_cell_10/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_5/GreaterEqual?
lstm_cell_10/dropout_5/CastCast'lstm_cell_10/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_5/Cast?
lstm_cell_10/dropout_5/Mul_1Mullstm_cell_10/dropout_5/Mul:z:0lstm_cell_10/dropout_5/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_5/Mul_1?
lstm_cell_10/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_6/Const?
lstm_cell_10/dropout_6/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_6/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_6/Mul?
lstm_cell_10/dropout_6/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_6/Shape?
3lstm_cell_10/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_6/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???25
3lstm_cell_10/dropout_6/random_uniform/RandomUniform?
%lstm_cell_10/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_6/GreaterEqual/y?
#lstm_cell_10/dropout_6/GreaterEqualGreaterEqual<lstm_cell_10/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_6/GreaterEqual?
lstm_cell_10/dropout_6/CastCast'lstm_cell_10/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_6/Cast?
lstm_cell_10/dropout_6/Mul_1Mullstm_cell_10/dropout_6/Mul:z:0lstm_cell_10/dropout_6/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_6/Mul_1?
lstm_cell_10/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/dropout_7/Const?
lstm_cell_10/dropout_7/MulMul!lstm_cell_10/ones_like_1:output:0%lstm_cell_10/dropout_7/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_7/Mul?
lstm_cell_10/dropout_7/ShapeShape!lstm_cell_10/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_10/dropout_7/Shape?
3lstm_cell_10/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_10/dropout_7/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2렫25
3lstm_cell_10/dropout_7/random_uniform/RandomUniform?
%lstm_cell_10/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%lstm_cell_10/dropout_7/GreaterEqual/y?
#lstm_cell_10/dropout_7/GreaterEqualGreaterEqual<lstm_cell_10/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_10/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2%
#lstm_cell_10/dropout_7/GreaterEqual?
lstm_cell_10/dropout_7/CastCast'lstm_cell_10/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_7/Cast?
lstm_cell_10/dropout_7/Mul_1Mullstm_cell_10/dropout_7/Mul:z:0lstm_cell_10/dropout_7/Cast:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/dropout_7/Mul_1?
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul?
lstm_cell_10/mul_1Mulstrided_slice_2:output:0 lstm_cell_10/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_1?
lstm_cell_10/mul_2Mulstrided_slice_2:output:0 lstm_cell_10/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_2?
lstm_cell_10/mul_3Mulstrided_slice_2:output:0 lstm_cell_10/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02#
!lstm_cell_10/split/ReadVariableOp?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split?
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul?
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_1?
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_2?
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1?
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dim?
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp?
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd?
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_1?
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_2?
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_3?
lstm_cell_10/mul_4Mulzeros:output:0 lstm_cell_10/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_4?
lstm_cell_10/mul_5Mulzeros:output:0 lstm_cell_10/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_5?
lstm_cell_10/mul_6Mulzeros:output:0 lstm_cell_10/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_6?
lstm_cell_10/mul_7Mulzeros:output:0 lstm_cell_10/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_7?
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp?
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack?
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1?
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2?
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice?
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_4?
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid?
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_1?
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack?
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1?
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2?
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1?
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_5?
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_8?
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_2?
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack?
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1?
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2?
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2?
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_6?
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh?
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_9?
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_3?
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_3?
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack?
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1?
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2?
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3?
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_7?
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_4?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh_1?
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_328431*
condR
while_cond_328430*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????&
"
_user_specified_name
inputs/0
??
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_328910
inputs_0.
*lstm_cell_10_split_readvariableop_resource0
,lstm_cell_10_split_1_readvariableop_resource(
$lstm_cell_10_readvariableop_resource
identity??:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?lstm_cell_10/ReadVariableOp?lstm_cell_10/ReadVariableOp_1?lstm_cell_10/ReadVariableOp_2?lstm_cell_10/ReadVariableOp_3?!lstm_cell_10/split/ReadVariableOp?#lstm_cell_10/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2
strided_slice_2?
lstm_cell_10/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_10/ones_like/Shape?
lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_10/ones_like/Const?
lstm_cell_10/ones_likeFill%lstm_cell_10/ones_like/Shape:output:0%lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/ones_like~
lstm_cell_10/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2 
lstm_cell_10/ones_like_1/Shape?
lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm_cell_10/ones_like_1/Const?
lstm_cell_10/ones_like_1Fill'lstm_cell_10/ones_like_1/Shape:output:0'lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/ones_like_1?
lstm_cell_10/mulMulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul?
lstm_cell_10/mul_1Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_1?
lstm_cell_10/mul_2Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_2?
lstm_cell_10/mul_3Mulstrided_slice_2:output:0lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2
lstm_cell_10/mul_3j
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
!lstm_cell_10/split/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02#
!lstm_cell_10/split/ReadVariableOp?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0)lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
lstm_cell_10/split?
lstm_cell_10/MatMulMatMullstm_cell_10/mul:z:0lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul?
lstm_cell_10/MatMul_1MatMullstm_cell_10/mul_1:z:0lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_1?
lstm_cell_10/MatMul_2MatMullstm_cell_10/mul_2:z:0lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_2?
lstm_cell_10/MatMul_3MatMullstm_cell_10/mul_3:z:0lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_3n
lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const_1?
lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
lstm_cell_10/split_1/split_dim?
#lstm_cell_10/split_1/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/split_1/ReadVariableOp?
lstm_cell_10/split_1Split'lstm_cell_10/split_1/split_dim:output:0+lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2
lstm_cell_10/split_1?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/MatMul:product:0lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd?
lstm_cell_10/BiasAdd_1BiasAddlstm_cell_10/MatMul_1:product:0lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_1?
lstm_cell_10/BiasAdd_2BiasAddlstm_cell_10/MatMul_2:product:0lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_2?
lstm_cell_10/BiasAdd_3BiasAddlstm_cell_10/MatMul_3:product:0lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/BiasAdd_3?
lstm_cell_10/mul_4Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_4?
lstm_cell_10/mul_5Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_5?
lstm_cell_10/mul_6Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_6?
lstm_cell_10/mul_7Mulzeros:output:0!lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_7?
lstm_cell_10/ReadVariableOpReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp?
 lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 lstm_cell_10/strided_slice/stack?
"lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice/stack_1?
"lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"lstm_cell_10/strided_slice/stack_2?
lstm_cell_10/strided_sliceStridedSlice#lstm_cell_10/ReadVariableOp:value:0)lstm_cell_10/strided_slice/stack:output:0+lstm_cell_10/strided_slice/stack_1:output:0+lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice?
lstm_cell_10/MatMul_4MatMullstm_cell_10/mul_4:z:0#lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_4?
lstm_cell_10/addAddV2lstm_cell_10/BiasAdd:output:0lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add
lstm_cell_10/SigmoidSigmoidlstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid?
lstm_cell_10/ReadVariableOp_1ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_1?
"lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm_cell_10/strided_slice_1/stack?
$lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_cell_10/strided_slice_1/stack_1?
$lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_1/stack_2?
lstm_cell_10/strided_slice_1StridedSlice%lstm_cell_10/ReadVariableOp_1:value:0+lstm_cell_10/strided_slice_1/stack:output:0-lstm_cell_10/strided_slice_1/stack_1:output:0-lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_1?
lstm_cell_10/MatMul_5MatMullstm_cell_10/mul_5:z:0%lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_5?
lstm_cell_10/add_1AddV2lstm_cell_10/BiasAdd_1:output:0lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mul_8Mullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_8?
lstm_cell_10/ReadVariableOp_2ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_2?
"lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"lstm_cell_10/strided_slice_2/stack?
$lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2&
$lstm_cell_10/strided_slice_2/stack_1?
$lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_2/stack_2?
lstm_cell_10/strided_slice_2StridedSlice%lstm_cell_10/ReadVariableOp_2:value:0+lstm_cell_10/strided_slice_2/stack:output:0-lstm_cell_10/strided_slice_2/stack_1:output:0-lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_2?
lstm_cell_10/MatMul_6MatMullstm_cell_10/mul_6:z:0%lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_6?
lstm_cell_10/add_2AddV2lstm_cell_10/BiasAdd_2:output:0lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_2x
lstm_cell_10/TanhTanhlstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh?
lstm_cell_10/mul_9Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_9?
lstm_cell_10/add_3AddV2lstm_cell_10/mul_8:z:0lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_3?
lstm_cell_10/ReadVariableOp_3ReadVariableOp$lstm_cell_10_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
lstm_cell_10/ReadVariableOp_3?
"lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2$
"lstm_cell_10/strided_slice_3/stack?
$lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_cell_10/strided_slice_3/stack_1?
$lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm_cell_10/strided_slice_3/stack_2?
lstm_cell_10/strided_slice_3StridedSlice%lstm_cell_10/ReadVariableOp_3:value:0+lstm_cell_10/strided_slice_3/stack:output:0-lstm_cell_10/strided_slice_3/stack_1:output:0-lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
lstm_cell_10/strided_slice_3?
lstm_cell_10/MatMul_7MatMullstm_cell_10/mul_7:z:0%lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/MatMul_7?
lstm_cell_10/add_4AddV2lstm_cell_10/BiasAdd_3:output:0lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/add_4?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Tanh_1Tanhlstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/Tanh_1?
lstm_cell_10/mul_10Mullstm_cell_10/Sigmoid_2:y:0lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_10/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_10_split_readvariableop_resource,lstm_cell_10_split_1_readvariableop_resource$lstm_cell_10_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_328762*
condR
while_cond_328761*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_10_split_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp,lstm_cell_10_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_10/ReadVariableOp^lstm_cell_10/ReadVariableOp_1^lstm_cell_10/ReadVariableOp_2^lstm_cell_10/ReadVariableOp_3"^lstm_cell_10/split/ReadVariableOp$^lstm_cell_10/split_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2:
lstm_cell_10/ReadVariableOplstm_cell_10/ReadVariableOp2>
lstm_cell_10/ReadVariableOp_1lstm_cell_10/ReadVariableOp_12>
lstm_cell_10/ReadVariableOp_2lstm_cell_10/ReadVariableOp_22>
lstm_cell_10/ReadVariableOp_3lstm_cell_10/ReadVariableOp_32F
!lstm_cell_10/split/ReadVariableOp!lstm_cell_10/split/ReadVariableOp2J
#lstm_cell_10/split_1/ReadVariableOp#lstm_cell_10/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????&
"
_user_specified_name
inputs/0
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_327188

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
while_cond_326214
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_326214___redundant_placeholder04
0while_while_cond_326214___redundant_placeholder14
0while_while_cond_326214___redundant_placeholder24
0while_while_cond_326214___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
while_cond_328761
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_328761___redundant_placeholder04
0while_while_cond_328761___redundant_placeholder14
0while_while_cond_328761___redundant_placeholder24
0while_while_cond_328761___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
??
?
%sequential_5_lstm_5_while_body_325443D
@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counterJ
Fsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations)
%sequential_5_lstm_5_while_placeholder+
'sequential_5_lstm_5_while_placeholder_1+
'sequential_5_lstm_5_while_placeholder_2+
'sequential_5_lstm_5_while_placeholder_3C
?sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1_0
{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0J
Fsequential_5_lstm_5_while_lstm_cell_10_split_readvariableop_resource_0L
Hsequential_5_lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0D
@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0&
"sequential_5_lstm_5_while_identity(
$sequential_5_lstm_5_while_identity_1(
$sequential_5_lstm_5_while_identity_2(
$sequential_5_lstm_5_while_identity_3(
$sequential_5_lstm_5_while_identity_4(
$sequential_5_lstm_5_while_identity_5A
=sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1}
ysequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensorH
Dsequential_5_lstm_5_while_lstm_cell_10_split_readvariableop_resourceJ
Fsequential_5_lstm_5_while_lstm_cell_10_split_1_readvariableop_resourceB
>sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource??5sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp?7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_1?7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_2?7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3?;sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp?=sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp?
Ksequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   2M
Ksequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_lstm_5_while_placeholderTsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype02?
=sequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem?
6sequential_5/lstm_5/while/lstm_cell_10/ones_like/ShapeShapeDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:28
6sequential_5/lstm_5/while/lstm_cell_10/ones_like/Shape?
6sequential_5/lstm_5/while/lstm_cell_10/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??28
6sequential_5/lstm_5/while/lstm_cell_10/ones_like/Const?
0sequential_5/lstm_5/while/lstm_cell_10/ones_likeFill?sequential_5/lstm_5/while/lstm_cell_10/ones_like/Shape:output:0?sequential_5/lstm_5/while/lstm_cell_10/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&22
0sequential_5/lstm_5/while/lstm_cell_10/ones_like?
8sequential_5/lstm_5/while/lstm_cell_10/ones_like_1/ShapeShape'sequential_5_lstm_5_while_placeholder_2*
T0*
_output_shapes
:2:
8sequential_5/lstm_5/while/lstm_cell_10/ones_like_1/Shape?
8sequential_5/lstm_5/while/lstm_cell_10/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2:
8sequential_5/lstm_5/while/lstm_cell_10/ones_like_1/Const?
2sequential_5/lstm_5/while/lstm_cell_10/ones_like_1FillAsequential_5/lstm_5/while/lstm_cell_10/ones_like_1/Shape:output:0Asequential_5/lstm_5/while/lstm_cell_10/ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 24
2sequential_5/lstm_5/while/lstm_cell_10/ones_like_1?
*sequential_5/lstm_5/while/lstm_cell_10/mulMulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_5/lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2,
*sequential_5/lstm_5/while/lstm_cell_10/mul?
,sequential_5/lstm_5/while/lstm_cell_10/mul_1MulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_5/lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_1?
,sequential_5/lstm_5/while/lstm_cell_10/mul_2MulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_5/lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_2?
,sequential_5/lstm_5/while/lstm_cell_10/mul_3MulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_5/lstm_5/while/lstm_cell_10/ones_like:output:0*
T0*'
_output_shapes
:?????????&2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_3?
,sequential_5/lstm_5/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_5/lstm_5/while/lstm_cell_10/Const?
6sequential_5/lstm_5/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential_5/lstm_5/while/lstm_cell_10/split/split_dim?
;sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOpReadVariableOpFsequential_5_lstm_5_while_lstm_cell_10_split_readvariableop_resource_0*
_output_shapes
:	&?*
dtype02=
;sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp?
,sequential_5/lstm_5/while/lstm_cell_10/splitSplit?sequential_5/lstm_5/while/lstm_cell_10/split/split_dim:output:0Csequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2.
,sequential_5/lstm_5/while/lstm_cell_10/split?
-sequential_5/lstm_5/while/lstm_cell_10/MatMulMatMul.sequential_5/lstm_5/while/lstm_cell_10/mul:z:05sequential_5/lstm_5/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:????????? 2/
-sequential_5/lstm_5/while/lstm_cell_10/MatMul?
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_1MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_1:z:05sequential_5/lstm_5/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:????????? 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_1?
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_2MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_2:z:05sequential_5/lstm_5/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:????????? 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_2?
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_3MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_3:z:05sequential_5/lstm_5/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:????????? 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_3?
.sequential_5/lstm_5/while/lstm_cell_10/Const_1Const*
_output_shapes
: *
dtype0*
value	B :20
.sequential_5/lstm_5/while/lstm_cell_10/Const_1?
8sequential_5/lstm_5/while/lstm_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8sequential_5/lstm_5/while/lstm_cell_10/split_1/split_dim?
=sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOpReadVariableOpHsequential_5_lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02?
=sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp?
.sequential_5/lstm_5/while/lstm_cell_10/split_1SplitAsequential_5/lstm_5/while/lstm_cell_10/split_1/split_dim:output:0Esequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split20
.sequential_5/lstm_5/while/lstm_cell_10/split_1?
.sequential_5/lstm_5/while/lstm_cell_10/BiasAddBiasAdd7sequential_5/lstm_5/while/lstm_cell_10/MatMul:product:07sequential_5/lstm_5/while/lstm_cell_10/split_1:output:0*
T0*'
_output_shapes
:????????? 20
.sequential_5/lstm_5/while/lstm_cell_10/BiasAdd?
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_1BiasAdd9sequential_5/lstm_5/while/lstm_cell_10/MatMul_1:product:07sequential_5/lstm_5/while/lstm_cell_10/split_1:output:1*
T0*'
_output_shapes
:????????? 22
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_1?
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_2BiasAdd9sequential_5/lstm_5/while/lstm_cell_10/MatMul_2:product:07sequential_5/lstm_5/while/lstm_cell_10/split_1:output:2*
T0*'
_output_shapes
:????????? 22
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_2?
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_3BiasAdd9sequential_5/lstm_5/while/lstm_cell_10/MatMul_3:product:07sequential_5/lstm_5/while/lstm_cell_10/split_1:output:3*
T0*'
_output_shapes
:????????? 22
0sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_3?
,sequential_5/lstm_5/while/lstm_cell_10/mul_4Mul'sequential_5_lstm_5_while_placeholder_2;sequential_5/lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_4?
,sequential_5/lstm_5/while/lstm_cell_10/mul_5Mul'sequential_5_lstm_5_while_placeholder_2;sequential_5/lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_5?
,sequential_5/lstm_5/while/lstm_cell_10/mul_6Mul'sequential_5_lstm_5_while_placeholder_2;sequential_5/lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_6?
,sequential_5/lstm_5/while/lstm_cell_10/mul_7Mul'sequential_5_lstm_5_while_placeholder_2;sequential_5/lstm_5/while/lstm_cell_10/ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_7?
5sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOpReadVariableOp@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype027
5sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp?
:sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack?
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_1?
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_2?
4sequential_5/lstm_5/while/lstm_cell_10/strided_sliceStridedSlice=sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp:value:0Csequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack:output:0Esequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_1:output:0Esequential_5/lstm_5/while/lstm_cell_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask26
4sequential_5/lstm_5/while/lstm_cell_10/strided_slice?
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_4MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_4:z:0=sequential_5/lstm_5/while/lstm_cell_10/strided_slice:output:0*
T0*'
_output_shapes
:????????? 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_4?
*sequential_5/lstm_5/while/lstm_cell_10/addAddV27sequential_5/lstm_5/while/lstm_cell_10/BiasAdd:output:09sequential_5/lstm_5/while/lstm_cell_10/MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2,
*sequential_5/lstm_5/while/lstm_cell_10/add?
.sequential_5/lstm_5/while/lstm_cell_10/SigmoidSigmoid.sequential_5/lstm_5/while/lstm_cell_10/add:z:0*
T0*'
_output_shapes
:????????? 20
.sequential_5/lstm_5/while/lstm_cell_10/Sigmoid?
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_1ReadVariableOp@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype029
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_1?
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2>
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack?
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_1?
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_2?
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1StridedSlice?sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_1:value:0Esequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_1:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask28
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1?
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_5MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_5:z:0?sequential_5/lstm_5/while/lstm_cell_10/strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_5?
,sequential_5/lstm_5/while/lstm_cell_10/add_1AddV29sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_1:output:09sequential_5/lstm_5/while/lstm_cell_10/MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/while/lstm_cell_10/add_1?
0sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_1Sigmoid0sequential_5/lstm_5/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:????????? 22
0sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_1?
,sequential_5/lstm_5/while/lstm_cell_10/mul_8Mul4sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_1:y:0'sequential_5_lstm_5_while_placeholder_3*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_8?
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_2ReadVariableOp@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype029
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_2?
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2>
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack?
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_1?
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_2?
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2StridedSlice?sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_2:value:0Esequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_1:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask28
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2?
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_6MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_6:z:0?sequential_5/lstm_5/while/lstm_cell_10/strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_6?
,sequential_5/lstm_5/while/lstm_cell_10/add_2AddV29sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_2:output:09sequential_5/lstm_5/while/lstm_cell_10/MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/while/lstm_cell_10/add_2?
+sequential_5/lstm_5/while/lstm_cell_10/TanhTanh0sequential_5/lstm_5/while/lstm_cell_10/add_2:z:0*
T0*'
_output_shapes
:????????? 2-
+sequential_5/lstm_5/while/lstm_cell_10/Tanh?
,sequential_5/lstm_5/while/lstm_cell_10/mul_9Mul2sequential_5/lstm_5/while/lstm_cell_10/Sigmoid:y:0/sequential_5/lstm_5/while/lstm_cell_10/Tanh:y:0*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/while/lstm_cell_10/mul_9?
,sequential_5/lstm_5/while/lstm_cell_10/add_3AddV20sequential_5/lstm_5/while/lstm_cell_10/mul_8:z:00sequential_5/lstm_5/while/lstm_cell_10/mul_9:z:0*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/while/lstm_cell_10/add_3?
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3ReadVariableOp@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype029
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3?
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2>
<sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack?
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_1?
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_2?
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3StridedSlice?sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3:value:0Esequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_1:output:0Gsequential_5/lstm_5/while/lstm_cell_10/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask28
6sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3?
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_7MatMul0sequential_5/lstm_5/while/lstm_cell_10/mul_7:z:0?sequential_5/lstm_5/while/lstm_cell_10/strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 21
/sequential_5/lstm_5/while/lstm_cell_10/MatMul_7?
,sequential_5/lstm_5/while/lstm_cell_10/add_4AddV29sequential_5/lstm_5/while/lstm_cell_10/BiasAdd_3:output:09sequential_5/lstm_5/while/lstm_cell_10/MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2.
,sequential_5/lstm_5/while/lstm_cell_10/add_4?
0sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_2Sigmoid0sequential_5/lstm_5/while/lstm_cell_10/add_4:z:0*
T0*'
_output_shapes
:????????? 22
0sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_2?
-sequential_5/lstm_5/while/lstm_cell_10/Tanh_1Tanh0sequential_5/lstm_5/while/lstm_cell_10/add_3:z:0*
T0*'
_output_shapes
:????????? 2/
-sequential_5/lstm_5/while/lstm_cell_10/Tanh_1?
-sequential_5/lstm_5/while/lstm_cell_10/mul_10Mul4sequential_5/lstm_5/while/lstm_cell_10/Sigmoid_2:y:01sequential_5/lstm_5/while/lstm_cell_10/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2/
-sequential_5/lstm_5/while/lstm_cell_10/mul_10?
>sequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_lstm_5_while_placeholder_1%sequential_5_lstm_5_while_placeholder1sequential_5/lstm_5/while/lstm_cell_10/mul_10:z:0*
_output_shapes
: *
element_dtype02@
>sequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem?
sequential_5/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_5/lstm_5/while/add/y?
sequential_5/lstm_5/while/addAddV2%sequential_5_lstm_5_while_placeholder(sequential_5/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_5/while/add?
!sequential_5/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_5/lstm_5/while/add_1/y?
sequential_5/lstm_5/while/add_1AddV2@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counter*sequential_5/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_5/while/add_1?
"sequential_5/lstm_5/while/IdentityIdentity#sequential_5/lstm_5/while/add_1:z:06^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2$
"sequential_5/lstm_5/while/Identity?
$sequential_5/lstm_5/while/Identity_1IdentityFsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations6^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_5/lstm_5/while/Identity_1?
$sequential_5/lstm_5/while/Identity_2Identity!sequential_5/lstm_5/while/add:z:06^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_5/lstm_5/while/Identity_2?
$sequential_5/lstm_5/while/Identity_3IdentityNsequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:06^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential_5/lstm_5/while/Identity_3?
$sequential_5/lstm_5/while/Identity_4Identity1sequential_5/lstm_5/while/lstm_cell_10/mul_10:z:06^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2&
$sequential_5/lstm_5/while/Identity_4?
$sequential_5/lstm_5/while/Identity_5Identity0sequential_5/lstm_5/while/lstm_cell_10/add_3:z:06^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp8^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_18^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_28^sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_3<^sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp>^sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2&
$sequential_5/lstm_5/while/Identity_5"Q
"sequential_5_lstm_5_while_identity+sequential_5/lstm_5/while/Identity:output:0"U
$sequential_5_lstm_5_while_identity_1-sequential_5/lstm_5/while/Identity_1:output:0"U
$sequential_5_lstm_5_while_identity_2-sequential_5/lstm_5/while/Identity_2:output:0"U
$sequential_5_lstm_5_while_identity_3-sequential_5/lstm_5/while/Identity_3:output:0"U
$sequential_5_lstm_5_while_identity_4-sequential_5/lstm_5/while/Identity_4:output:0"U
$sequential_5_lstm_5_while_identity_5-sequential_5/lstm_5/while/Identity_5:output:0"?
>sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource@sequential_5_lstm_5_while_lstm_cell_10_readvariableop_resource_0"?
Fsequential_5_lstm_5_while_lstm_cell_10_split_1_readvariableop_resourceHsequential_5_lstm_5_while_lstm_cell_10_split_1_readvariableop_resource_0"?
Dsequential_5_lstm_5_while_lstm_cell_10_split_readvariableop_resourceFsequential_5_lstm_5_while_lstm_cell_10_split_readvariableop_resource_0"?
=sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1?sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1_0"?
ysequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2n
5sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp5sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp2r
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_17sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_12r
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_27sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_22r
7sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_37sequential_5/lstm_5/while/lstm_cell_10/ReadVariableOp_32z
;sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp;sequential_5/lstm_5/while/lstm_cell_10/split/ReadVariableOp2~
=sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp=sequential_5/lstm_5/while/lstm_cell_10/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_327239

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_329694

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
*__inference_dropout_5_layer_call_fn_329658

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_3271832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_lstm_cell_10_layer_call_fn_330005

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_3258972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????&:????????? :????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?
?
while_cond_326965
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_326965___redundant_placeholder04
0while_while_cond_326965___redundant_placeholder14
0while_while_cond_326965___redundant_placeholder24
0while_while_cond_326965___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?%
?
while_body_326359
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_10_326383_0
while_lstm_cell_10_326385_0
while_lstm_cell_10_326387_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_10_326383
while_lstm_cell_10_326385
while_lstm_cell_10_326387??*while/lstm_cell_10/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????&*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_326383_0while_lstm_cell_10_326385_0while_lstm_cell_10_326387_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_3258972,
*while/lstm_cell_10/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_10_326383while_lstm_cell_10_326383_0"8
while_lstm_cell_10_326385while_lstm_cell_10_326385_0"8
while_lstm_cell_10_326387while_lstm_cell_10_326387_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?g
?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_329971

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????&2
	ones_like^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:?????????&2
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????&2
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????&2
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????&2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:& :& :& :& *
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:????????? 2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:????????? 2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:????????? 2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:????????? 2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
: : : : *
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:????????? 2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:????????? 2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:????????? 2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:????????? 2
	BiasAdd_3g
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
mul_4g
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
mul_5g
mul_6Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
mul_6g
mul_7Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:????????? 2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:????????? 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:????????? 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:????????? 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:????????? 2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:????????? 2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:????????? 2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	 ?*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:????????? 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:????????? 2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
mul_10?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????&:????????? :????????? :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_329653

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
while_cond_326358
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_326358___redundant_placeholder04
0while_while_cond_326358___redundant_placeholder14
0while_while_cond_326358___redundant_placeholder24
0while_while_cond_326358___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?	
?
D__inference_dense_16_layer_call_and_return_conditional_losses_327212

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?[
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_326440

inputs
lstm_cell_10_326346
lstm_cell_10_326348
lstm_cell_10_326350
identity??:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?$lstm_cell_10/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????&2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????&   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????&*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_326346lstm_cell_10_326348lstm_cell_10_326350*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_3258972&
$lstm_cell_10/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_326346lstm_cell_10_326348lstm_cell_10_326350*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_326359*
condR
while_cond_326358*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_10_326346*
_output_shapes
:	&?*
dtype02>
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp?
-lstm_5/lstm_cell_10/kernel/Regularizer/SquareSquareDlstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	&?2/
-lstm_5/lstm_cell_10/kernel/Regularizer/Square?
,lstm_5/lstm_cell_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_5/lstm_cell_10/kernel/Regularizer/Const?
*lstm_5/lstm_cell_10/kernel/Regularizer/SumSum1lstm_5/lstm_cell_10/kernel/Regularizer/Square:y:05lstm_5/lstm_cell_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/Sum?
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,lstm_5/lstm_cell_10/kernel/Regularizer/mul/x?
*lstm_5/lstm_cell_10/kernel/Regularizer/mulMul5lstm_5/lstm_cell_10/kernel/Regularizer/mul/x:output:03lstm_5/lstm_cell_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_5/lstm_cell_10/kernel/Regularizer/mul?
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_10_326348*
_output_shapes	
:?*
dtype02<
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp?
+lstm_5/lstm_cell_10/bias/Regularizer/SquareSquareBlstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2-
+lstm_5/lstm_cell_10/bias/Regularizer/Square?
*lstm_5/lstm_cell_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*lstm_5/lstm_cell_10/bias/Regularizer/Const?
(lstm_5/lstm_cell_10/bias/Regularizer/SumSum/lstm_5/lstm_cell_10/bias/Regularizer/Square:y:03lstm_5/lstm_cell_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/Sum?
*lstm_5/lstm_cell_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2,
*lstm_5/lstm_cell_10/bias/Regularizer/mul/x?
(lstm_5/lstm_cell_10/bias/Regularizer/mulMul3lstm_5/lstm_cell_10/bias/Regularizer/mul/x:output:01lstm_5/lstm_cell_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(lstm_5/lstm_cell_10/bias/Regularizer/mul?
IdentityIdentitystrided_slice_3:output:0;^lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp=^lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp%^lstm_cell_10/StatefulPartitionedCall^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????&:::2x
:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp:lstm_5/lstm_cell_10/bias/Regularizer/Square/ReadVariableOp2|
<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp<lstm_5/lstm_cell_10/kernel/Regularizer/Square/ReadVariableOp2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????&
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
lstm_5_input9
serving_default_lstm_5_input:0?????????&<
dense_170
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?5
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
u__call__
v_default_save_signature
*w&call_and_return_all_conditional_losses"?2
_tf_keras_sequential?2{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_5_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "recurrent_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 38]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 38]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_5_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "recurrent_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"?
_tf_keras_rnn_layer?{"class_name": "LSTM", "name": "lstm_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 38]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "recurrent_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 38]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 38]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
z__call__
*{&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
|__call__
*}&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
?

kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
~__call__
*&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
(iter

)beta_1

*beta_2
	+decay
,learning_ratemcmdmemf"mg#mh-mi.mj/mkvlvmvnvo"vp#vq-vr.vs/vt"
	optimizer
_
-0
.1
/2
3
4
5
6
"7
#8"
trackable_list_wrapper
_
-0
.1
/2
3
4
5
6
"7
#8"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
0layer_regularization_losses
1non_trainable_variables
2metrics
	regularization_losses
3layer_metrics

4layers
u__call__
v_default_save_signature
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?	

-kernel
.recurrent_kernel
/bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "recurrent_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}}
 "
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
trainable_variables
	variables
9layer_regularization_losses
:non_trainable_variables
;metrics
regularization_losses

<states
=layer_metrics

>layers
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_15/kernel
: 2dense_15/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
?layer_regularization_losses
@non_trainable_variables
Ametrics
regularization_losses
Blayer_metrics

Clayers
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Dlayer_regularization_losses
Enon_trainable_variables
Fmetrics
regularization_losses
Glayer_metrics

Hlayers
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_16/kernel
: 2dense_16/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Ilayer_regularization_losses
Jnon_trainable_variables
Kmetrics
 regularization_losses
Llayer_metrics

Mlayers
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_17/kernel
:2dense_17/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
$trainable_variables
%	variables
Nlayer_regularization_losses
Onon_trainable_variables
Pmetrics
&regularization_losses
Qlayer_metrics

Rlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2iter
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
-:+	&?2lstm_5/lstm_cell_10/kernel
7:5	 ?2$lstm_5/lstm_cell_10/recurrent_kernel
':%?2lstm_5/lstm_cell_10/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
5trainable_variables
6	variables
Ulayer_regularization_losses
Vnon_trainable_variables
Wmetrics
7regularization_losses
Xlayer_metrics

Ylayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	Ztotal
	[count
\	variables
]	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
Z0
[1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
^0
_1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
!:  2dense_15/kernel/m
: 2dense_15/bias/m
!:  2dense_16/kernel/m
: 2dense_16/bias/m
!: 2dense_17/kernel/m
:2dense_17/bias/m
-:+	&?2lstm_5/lstm_cell_10/kernel/m
7:5	 ?2&lstm_5/lstm_cell_10/recurrent_kernel/m
':%?2lstm_5/lstm_cell_10/bias/m
!:  2dense_15/kernel/v
: 2dense_15/bias/v
!:  2dense_16/kernel/v
: 2dense_16/bias/v
!: 2dense_17/kernel/v
:2dense_17/bias/v
-:+	&?2lstm_5/lstm_cell_10/kernel/v
7:5	 ?2&lstm_5/lstm_cell_10/recurrent_kernel/v
':%?2lstm_5/lstm_cell_10/bias/v
?2?
-__inference_sequential_5_layer_call_fn_327370
-__inference_sequential_5_layer_call_fn_328213
-__inference_sequential_5_layer_call_fn_327432
-__inference_sequential_5_layer_call_fn_328236?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_325601?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
lstm_5_input?????????&
?2?
H__inference_sequential_5_layer_call_and_return_conditional_losses_327268
H__inference_sequential_5_layer_call_and_return_conditional_losses_328190
H__inference_sequential_5_layer_call_and_return_conditional_losses_327307
H__inference_sequential_5_layer_call_and_return_conditional_losses_327901?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_lstm_5_layer_call_fn_329616
'__inference_lstm_5_layer_call_fn_328932
'__inference_lstm_5_layer_call_fn_329605
'__inference_lstm_5_layer_call_fn_328921?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_lstm_5_layer_call_and_return_conditional_losses_328643
B__inference_lstm_5_layer_call_and_return_conditional_losses_329594
B__inference_lstm_5_layer_call_and_return_conditional_losses_328910
B__inference_lstm_5_layer_call_and_return_conditional_losses_329327?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_15_layer_call_fn_329636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_15_layer_call_and_return_conditional_losses_329627?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_5_layer_call_fn_329658
*__inference_dropout_5_layer_call_fn_329663?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_5_layer_call_and_return_conditional_losses_329653
E__inference_dropout_5_layer_call_and_return_conditional_losses_329648?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_16_layer_call_fn_329683?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_16_layer_call_and_return_conditional_losses_329674?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_17_layer_call_fn_329703?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_17_layer_call_and_return_conditional_losses_329694?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_327477lstm_5_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_lstm_cell_10_layer_call_fn_329988
-__inference_lstm_cell_10_layer_call_fn_330005?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_329875
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_329971?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_0_330016?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_330027?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? ?
!__inference__wrapped_model_325601{	-/."#9?6
/?,
*?'
lstm_5_input?????????&
? "3?0
.
dense_17"?
dense_17??????????
D__inference_dense_15_layer_call_and_return_conditional_losses_329627\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? |
)__inference_dense_15_layer_call_fn_329636O/?,
%?"
 ?
inputs????????? 
? "?????????? ?
D__inference_dense_16_layer_call_and_return_conditional_losses_329674\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? |
)__inference_dense_16_layer_call_fn_329683O/?,
%?"
 ?
inputs????????? 
? "?????????? ?
D__inference_dense_17_layer_call_and_return_conditional_losses_329694\"#/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? |
)__inference_dense_17_layer_call_fn_329703O"#/?,
%?"
 ?
inputs????????? 
? "???????????
E__inference_dropout_5_layer_call_and_return_conditional_losses_329648\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ?
E__inference_dropout_5_layer_call_and_return_conditional_losses_329653\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? }
*__inference_dropout_5_layer_call_fn_329658O3?0
)?&
 ?
inputs????????? 
p
? "?????????? }
*__inference_dropout_5_layer_call_fn_329663O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ;
__inference_loss_fn_0_330016-?

? 
? "? ;
__inference_loss_fn_1_330027/?

? 
? "? ?
B__inference_lstm_5_layer_call_and_return_conditional_losses_328643}-/.O?L
E?B
4?1
/?,
inputs/0??????????????????&

 
p

 
? "%?"
?
0????????? 
? ?
B__inference_lstm_5_layer_call_and_return_conditional_losses_328910}-/.O?L
E?B
4?1
/?,
inputs/0??????????????????&

 
p 

 
? "%?"
?
0????????? 
? ?
B__inference_lstm_5_layer_call_and_return_conditional_losses_329327m-/.??<
5?2
$?!
inputs?????????&

 
p

 
? "%?"
?
0????????? 
? ?
B__inference_lstm_5_layer_call_and_return_conditional_losses_329594m-/.??<
5?2
$?!
inputs?????????&

 
p 

 
? "%?"
?
0????????? 
? ?
'__inference_lstm_5_layer_call_fn_328921p-/.O?L
E?B
4?1
/?,
inputs/0??????????????????&

 
p

 
? "?????????? ?
'__inference_lstm_5_layer_call_fn_328932p-/.O?L
E?B
4?1
/?,
inputs/0??????????????????&

 
p 

 
? "?????????? ?
'__inference_lstm_5_layer_call_fn_329605`-/.??<
5?2
$?!
inputs?????????&

 
p

 
? "?????????? ?
'__inference_lstm_5_layer_call_fn_329616`-/.??<
5?2
$?!
inputs?????????&

 
p 

 
? "?????????? ?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_329875?-/.??}
v?s
 ?
inputs?????????&
K?H
"?
states/0????????? 
"?
states/1????????? 
p
? "s?p
i?f
?
0/0????????? 
E?B
?
0/1/0????????? 
?
0/1/1????????? 
? ?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_329971?-/.??}
v?s
 ?
inputs?????????&
K?H
"?
states/0????????? 
"?
states/1????????? 
p 
? "s?p
i?f
?
0/0????????? 
E?B
?
0/1/0????????? 
?
0/1/1????????? 
? ?
-__inference_lstm_cell_10_layer_call_fn_329988?-/.??}
v?s
 ?
inputs?????????&
K?H
"?
states/0????????? 
"?
states/1????????? 
p
? "c?`
?
0????????? 
A?>
?
1/0????????? 
?
1/1????????? ?
-__inference_lstm_cell_10_layer_call_fn_330005?-/.??}
v?s
 ?
inputs?????????&
K?H
"?
states/0????????? 
"?
states/1????????? 
p 
? "c?`
?
0????????? 
A?>
?
1/0????????? 
?
1/1????????? ?
H__inference_sequential_5_layer_call_and_return_conditional_losses_327268u	-/."#A?>
7?4
*?'
lstm_5_input?????????&
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_5_layer_call_and_return_conditional_losses_327307u	-/."#A?>
7?4
*?'
lstm_5_input?????????&
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_5_layer_call_and_return_conditional_losses_327901o	-/."#;?8
1?.
$?!
inputs?????????&
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_5_layer_call_and_return_conditional_losses_328190o	-/."#;?8
1?.
$?!
inputs?????????&
p 

 
? "%?"
?
0?????????
? ?
-__inference_sequential_5_layer_call_fn_327370h	-/."#A?>
7?4
*?'
lstm_5_input?????????&
p

 
? "???????????
-__inference_sequential_5_layer_call_fn_327432h	-/."#A?>
7?4
*?'
lstm_5_input?????????&
p 

 
? "???????????
-__inference_sequential_5_layer_call_fn_328213b	-/."#;?8
1?.
$?!
inputs?????????&
p

 
? "???????????
-__inference_sequential_5_layer_call_fn_328236b	-/."#;?8
1?.
$?!
inputs?????????&
p 

 
? "???????????
$__inference_signature_wrapper_327477?	-/."#I?F
? 
??<
:
lstm_5_input*?'
lstm_5_input?????????&"3?0
.
dense_17"?
dense_17?????????