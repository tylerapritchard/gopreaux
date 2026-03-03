Archive Structure
=================
.. mermaid::
   :name: archive
   
    flowchart TD
        A{Data};
        subgraph Types;
            A---->B[FBOT];
            A---->C[SESNe];
            A---->D[SLSN-I];
            A---->E[SLSN-II];
            A---->F[SNII];
            A---->G[SNIIn];
            A---->a[Other];
        end;
        subgraph Subtypes;
            B---->H(SNIbn);
            B---->I(SNIcn);
            C---->J(SNIIb);
            C---->V(SNIb);
            C---->W(SNIc);
            D---->K(SLSN-II);
            E---->L(SLSN-I);
            F---->M(SNIIP);
            F---->N(SNII-pec);
            F---->O(SNII);
            G---->P(SNIIn);
            a---->Q(SNIa);
            a---->R(SNIa-pec);
            a---->X(TDE);
            a---->Y(Unclassified/Other);
        end;
        S{Data}-->T[Type]-->U(Subtype);

        style H fill:#e0f8f8,stroke:#2E86C1
        style I fill:#e0f8f8,stroke:#2E86C1
        style J fill:#e0f8f8,stroke:#2E86C1
        style V fill:#e0f8f8,stroke:#2E86C1
        style W fill:#e0f8f8,stroke:#2E86C1
        style K fill:#e0f8f8,stroke:#2E86C1
        style L fill:#e0f8f8,stroke:#2E86C1
        style M fill:#e0f8f8,stroke:#2E86C1
        style N fill:#e0f8f8,stroke:#2E86C1
        style O fill:#e0f8f8,stroke:#2E86C1
        style P fill:#e0f8f8,stroke:#2E86C1
        style Q fill:#e0f8f8,stroke:#2E86C1
        style R fill:#e0f8f8,stroke:#2E86C1
        style X fill:#e0f8f8,stroke:#2E86C1
        style Y fill:#e0f8f8,stroke:#2E86C1
        style U fill:#e0f8f8,stroke:#2E86C1
        style A fill:#daf7a6,stroke:#1E8449
        style S fill:#daf7a6,stroke:#1E8449```

The above chart details the out-of-the-box organization of transient data into types and subtypes. The structure was chosen to group transients based on similar light curve and SED behavior, rather than physical characteristics such as powering mechanisms or progenitor channels.