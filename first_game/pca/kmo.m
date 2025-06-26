function kmo=kmo(SA)
    X=corrcoef(SA);%SA是样本的原始数据
    iX = inv(X);     %X是原始数据的相关系数矩阵R，而inv表示求X的逆矩阵iX
    S2 = diag(diag((iX.^-1)));    %将iX的对角线的元素取倒数，其余元素都变为0，得到矩阵S2
    AIS = S2*iX*S2;    %anti-image covariance matrix，即AIS是反映像协方差矩阵   
    IS = X+AIS-2*S2;    %image covariance matrix，即IS是映像协方差矩阵
    Dai = diag(diag(sqrt(AIS)));    %就是将矩阵AIS对角线上的元素开平方，并且将其余元素都变成0，得到矩阵Dai
    IR = inv(Dai)*IS*inv(Dai); %image correlation matrix，即IR是映像相关矩阵
    AIR = inv(Dai)*AIS*inv(Dai); %anti-image correlation matrix，即AIR是反映像相关矩阵
    a = sum((AIR - diag(diag(AIR))).^2);    %diag(diag(AIR))表示将矩阵AIR的对角线取出来，再构造成一个对角矩阵（即对角线之   外元素都是 0）；. 表示将偏相关系数矩阵AIR - diag(diag(AIR))的每一个元素乘方，这样得到矩阵a。
    AA = sum(a);              %得到偏相关系数矩阵AIR - diag(diag(AIR))中所有偏相关系数的平方和AA，但不考虑其对角线上的数值。
    b = sum((X - eye(size(X))).^2);    %eye（）是单位矩阵；b就是将相关系数矩阵R中每一个元素乘方，但R对角线元素全部变成0
    BB = sum(b);             %BB就是所有变量之间（不包括变量自己与自己）的相关系数的平方和。
    kmo = BB/(AA+BB);   %KMO就是所有变量之间相关系数的平方和除以它与所有变量之间偏相关系数平方和的商，但不考虑变量  自己与自己的相关系数1以及偏相关系数。
end
