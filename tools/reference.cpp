// Sequential CPU version of the wildfire rules, written to follow the problem
// statement line by line (no optimisation). Used to generate expected outputs
// for random test cases and check the GPU version against them.
//
// Build:  g++ -O2 -o reference reference.cpp
// Run:    ./reference input.in.bin expected.out.bin
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb");
    int h[11]; fread(h, 4, 11, f);
    int N=h[0],T=h[1],H=h[2],dr=h[3],dc=h[4],W=h[5],S=h[6],D=h[7],I=h[8],R=h[9],X=h[10];
    size_t NN = (size_t)N*N;
    vector<uint8_t> fuel(NN), elev(NN), st(NN, 0), station(NN, 0);
    fread(fuel.data(),1,NN,f); fread(elev.data(),1,NN,f);
    int F; fread(&F,4,1,f); vector<int> fires(2*F); fread(fires.data(),4,2*F,f);
    int K; fread(&K,4,1,f); vector<int> stn(2*K); fread(stn.data(),4,2*K,f);
    int C; fread(&C,4,1,f); vector<int> cr(3*C); fread(cr.data(),4,3*C,f);
    fclose(f);
    for (int i=0;i<F;i++) st[fires[2*i]*N+fires[2*i+1]] = 1;
    for (int i=0;i<K;i++) station[stn[2*i]*N+stn[2*i+1]] = 1;
    auto steps = [&](int r1,int c1,int r2,int c2){ int d=max(abs(r1-r2),abs(c1-c2)); return (d+R-1)/R; };
    for (int t=0;t<T;t++) {
        vector<uint8_t> G = st, nst = st, nfuel = fuel;
        for (int r=0;r<N;r++) for (int c=0;c<N;c++) {
            size_t i=(size_t)r*N+c;
            if (G[i]==1) { nfuel[i]--; if (nfuel[i]==0) nst[i]=2; continue; }
            if (G[i]!=0 || fuel[i]==0 || station[i]) continue;
            long long heat=0;
            for (int rs=r-2;rs<=r+2;rs++) for (int cs=c-2;cs<=c+2;cs++) {
                if (rs<0||cs<0||rs>=N||cs>=N) continue;
                if ((rs==r&&cs==c) || G[(size_t)rs*N+cs]!=1) continue;
                long long a=H,b=1; int orr=r-rs, oc=c-cs;
                if ((orr==dr&&oc==dc)||(orr==2*dr&&oc==2*dc)) a*=W;
                else if ((orr==-dr&&oc==-dc)||(orr==-2*dr&&oc==-2*dc)) b*=W;
                int del = elev[i]-elev[(size_t)rs*N+cs];
                if (del>D) a*=S; else if (del<-D) b*=S;
                heat += a/b;
            }
            if (heat>=I) nst[i]=1;
        }
        vector<char> wet(NN,0);
        for (int k=0;k<C;k++) {
            int &r=cr[3*k], &c=cr[3*k+1], &w=cr[3*k+2];
            int tr=-1,tc=-1;
            if (w>0) {
                int best=INT_MAX;
                for (int rr=0;rr<N;rr++) for (int cc=0;cc<N;cc++) if (G[(size_t)rr*N+cc]==1) {
                    int s=steps(r,c,rr,cc); if (s<best){best=s;tr=rr;tc=cc;}
                }
            } else {
                int best=INT_MAX;
                for (int j=0;j<K;j++){ int sr=stn[2*j],sc=stn[2*j+1]; int s=steps(r,c,sr,sc);
                    if (s<best || (s==best && (sr<tr || (sr==tr&&sc<tc)))){best=s;tr=sr;tc=sc;} }
            }
            if (tr>=0) {
                int mr=min(R,abs(tr-r)), mc=min(R,abs(tc-c));
                r += (tr>r?mr:-mr); c += (tc>c?mc:-mc);
            }
            size_t ci=(size_t)r*N+c;
            if (G[ci]==1 && w>0) { wet[ci]=1; w--; }
            if (station[ci]) w=X;
        }
        for (size_t i=0;i<NN;i++) if (wet[i]) nst[i]=3;
        st=nst; fuel=nfuel;
    }
    FILE* o=fopen(argv[2],"wb");
    fwrite(st.data(),1,NN,o); fwrite(fuel.data(),1,NN,o); fwrite(cr.data(),4,3*C,o); fclose(o);
}
