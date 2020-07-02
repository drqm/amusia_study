\layout {
  ragged-right = ##f
}
\markup {
\bold \sans A)
}

\new Staff \with {
  instrumentName = \markup {
  \center-column {\normalsize \sans "Low"
    }
  }
}
\relative c'' {
\key f \major
\tempo 4 = 120
c,8 c  
\override NoteHead.color = #red
c
\override NoteHead.color = #black
c c c c   
\override NoteHead.color = #(x11-color 'LimeGreen)
c 
\override NoteHead.color = #black
c c 
\override NoteHead.color = #(x11-color 'orange)
c 
\override NoteHead.color = #black
c c 
\override NoteHead.color = #blue 
c 
\override NoteHead.color = #black
c c c c
\override NoteHead.color = #magenta 
c
\override NoteHead.color = #black
c 
\override NoteHead.color = #red
c
\override NoteHead.color = #black
c c c c
\override NoteHead.color = #(x11-color 'orange)
c
\override NoteHead.color = #black 
c c c c
\override NoteHead.color = #(x11-color 'LimeGreen)
c
\override NoteHead.color = #black
c \bar "|."
}

\new Staff \with {
  instrumentName = \markup {
  \center-column {\normalsize \sans "Int."
    }
  }
}
\relative c'' {
\key f \major
f,8
\override NoteHead.color = #red
c'
\override NoteHead.color = #black
a c 
\override NoteHead.color = #magenta
f,
\override NoteHead.color = #black
c' a c f, c' a 
\override NoteHead.color = #(x11-color 'LimeGreen)
c 
\override NoteHead.color = #black
f, c' 
\override NoteHead.color = #blue
a 
\override NoteHead.color = #black
c
\override NoteHead.color = #(x11-color 'orange)
f, 
\override NoteHead.color = #black
c' a c f, 
\override NoteHead.color = #(x11-color 'LimeGreen)
c'
\override NoteHead.color = #black
a c f, c' 
\override NoteHead.color = #blue 
a 
\override NoteHead.color = #black
c 
\override NoteHead.color = #red
f, 
\override NoteHead.color = #black
c' a c \bar "|."
}

\new Staff \with {
  instrumentName = \markup {
  \center-column {\normalsize \sans "High"
   }
  }
}
\relative f {
\key f \major
f8 a 
\override NoteHead.color = #blue
c 
\override NoteHead.color = #black
f e f d
\override NoteHead.color = #(x11-color 'LimeGreen)
e 
\override NoteHead.color = #black
c 
\override NoteHead.color = #(x11-color 'orange)
f
\override NoteHead.color = #black
a c bes
\override NoteHead.color = #red
c 
\override NoteHead.color = #black
a bes 
\override NoteHead.color = #magenta
g
\override NoteHead.color = #black
f e d c
\override NoteHead.color = #blue 
bes
\override NoteHead.color = #black
a g f a g 
\override NoteHead.color = #magenta
e
\override NoteHead.color = #black
 f a 
\override NoteHead.color = #(x11-color 'orange)
c 
\override NoteHead.color = #black
a \bar "|." 
}

\markup {
\lower #6 \bold \sans
B)
}

\new Staff \with {
  instrumentName = \markup {
  \center-column {\normalsize \sans "Familiar"
    }
  }
}


\relative c'' {
\key g \major

g a
\override NoteHead.color = #red
b
\override NoteHead.color = #black
g 
\override NoteHead.color = #(x11-color 'orange)
a4
\override NoteHead.color = #black
a8 b c4 
\override NoteHead.color = #(x11-color 'LimeGreen)
c 
\override NoteHead.color = #black
b b g8 a 
\override NoteHead.color = #blue
b
\override NoteHead.color = #black
g a4 a8
\override NoteHead.color = #magenta
b 
\override NoteHead.color = #black
c4 d g,2 \bar "|."
}

\new Staff \with {
  instrumentName = \markup {
  \center-column {\normalsize \sans "Unfamiliar"
   }
  }
}


\relative c'' {
\key g \major
d4 c_\markup {
\lower #7 \bold \sans {
\with-color #red pitch
\hspace #6 
\with-color #blue intensity 
\hspace #6 
\with-color #(x11-color 'orange) timbre 
\hspace #6 
\with-color #(x11-color 'LimeGreen) location
\hspace #6 
\with-color #magenta rhythm
}
} 

b8
\override NoteHead.color = #(x11-color 'LimeGreen)
g 
\override NoteHead.color = #black
g b a
\override NoteHead.color = #(x11-color 'orange)
a4 
\override NoteHead.color = #black
a8 c4 c
\override NoteHead.color = #red
b
\override NoteHead.color = #black
g8 a8 b4 
\override NoteHead.color = #blue 
g8
\override NoteHead.color = #black
b a4
\override NoteHead.color = #magenta
a8
\override NoteHead.color = #black
b g2 \bar "|." 
}



