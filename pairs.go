// Copyright 2023 The Zero Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"
)

// Pairs is a language pair
type Pair struct {
	German  string
	English string
}

// https://strommeninc.com/1000-most-common-german-words-frequency-vocabulary/
var data string = `1 	wie 	as
2 	ich 	I
3 	seine 	his
4 	dass 	that
5 	er 	he
6 	war 	was
7 	für 	for
8 	auf 	on
9 	sind 	are
10 	mit 	with
11 	sie 	they
12 	sein 	be
13 	bei 	at
14 	ein 	one
15 	haben 	have
16 	dies 	this
17 	aus 	from
18 	durch 	by
19 	heiß 	hot
20 	Wort 	word
21 	aber 	but
22 	was 	what
23 	einige 	some
24 	ist 	is
25 	es 	it
26 	Sie 	you
27 	oder 	or
28 	hatte 	had
29 	die 	the
30 	von 	of
31 	zu 	to
32 	und 	and
33 	ein 	a
34 	bei 	in
35 	wir 	we
36 	können 	can
37 	aus 	out
38 	andere 	other
39 	waren 	were
40 	die 	which
41 	tun 	do
42 	ihre 	their
43 	Zeit 	time
44 	wenn 	if
45 	werden 	will
46 	wie 	how
47 	sagte 	said
48 	ein 	an
49 	jeder 	each
50 	sagen 	tell
51 	tut 	does
52 	Satz 	set
53 	drei 	three
54 	wollen 	want
55 	Luft 	air
56 	gut 	well
57 	auch 	also
58 	spielen 	play
59 	klein 	small
60 	Ende 	end
61 	setzen 	put
62 	Zuhause 	home
63 	lesen 	read
64 	seits 	hand
65 	Hafen 	port
66 	groß 	large
67 	buchstabieren 	spell
68 	hinzufügen 	add
69 	auch 	even
70 	Lande 	land
71 	hier 	here
72 	muss 	must
73 	groß 	big
74 	hoch 	high
75 	so 	such
76 	folgen 	follow
77 	Akt 	act
78 	warum 	why
79 	fragen 	ask
80 	Männer 	men
81 	Veränderung 	change
82 	ging 	went
83 	Licht 	light
84 	Art 	kind
85 	aus 	off
86 	müssen 	need
87 	Haus 	house
88 	Bild 	picture
89 	versuchen 	try
90 	uns 	us
91 	wieder 	again
92 	Tier 	animal
93 	Punkt 	point
94 	Mutter 	mother
95 	Welt 	world
96 	in der Nähe von 	near
97 	bauen 	build
98 	selbst 	self
99 	Erde 	earth
100 	Vater 	father
101 	jeder 	any
102 	neu 	new
103 	Arbeit 	work
104 	Teil 	part
105 	nehmen 	take
106 	erhalten 	get
107 	Ort 	place
108 	gemacht 	made
109 	leben 	live
110 	wo 	where
111 	nach 	after
112 	zurück 	back
113 	wenig 	little
114 	nur 	only
115 	Runde 	round
116 	Mann 	man
117 	Jahr 	year
118 	kam 	came
119 	zeigen 	show
120 	jeder 	every
121 	gut 	good
122 	mir 	me
123 	geben 	give
124 	unsere 	our
125 	unter 	under
126 	Name 	name
127 	sehr 	very
128 	durch 	through
129 	nur 	just
130 	Formular 	form
131 	Satz 	sentence
132 	groß 	great
133 	denken 	think
134 	sagen 	say
135 	Hilfe 	help
136 	niedrig 	low
137 	Linie 	line
138 	abweichen 	differ
139 	wiederum 	turn
140 	Ursache 	cause
141 	viel 	much
142 	bedeuten 	mean
143 	vor 	before
144 	Umzug 	move
145 	Recht 	right
146 	Junge 	boy
147 	alt 	old
148 	zu 	too
149 	gleich 	same
150 	sie 	she
151 	alle 	all
152 	da 	there
153 	wenn 	when
154 	nach oben 	up
155 	Verwendung 	use
156 	Ihre 	your
157 	Weg 	way
158 	über 	about
159 	viele 	many
160 	dann 	then
161 	sie 	them
162 	schreiben 	write
163 	würde 	would
164 	wie 	like
165 	so 	so
166 	diese 	these
167 	sie 	her
168 	lange 	long
169 	machen 	make
170 	Sache 	thing
171 	sehen 	see
172 	ihm 	him
173 	zwei 	two
174 	hat 	has
175 	suchen 	look
176 	mehr 	more
177 	Tag 	day
178 	könnte 	could
179 	gehen 	go
180 	kommen 	come
181 	tat 	did
182 	Anzahl 	number
183 	klingen 	sound
184 	nicht 	no
185 	am meisten 	most
186 	Menschen 	people
187 	meine 	my
188 	über 	over
189 	wissen 	know
190 	Wasser 	water
191 	als 	than
192 	Anruf 	call
193 	erste 	first
194 	die 	who
195 	können 	may
196 	nach unten 	down
197 	Seite 	side
198 	gewesen 	been
199 	jetzt 	now
200 	finden 	find
201 	Kopf 	head
202 	stehen 	stand
203 	besitzen 	own
204 	Seite 	page
205 	sollte 	should
206 	Land 	country
207 	gefunden 	found
208 	Antwort 	answer
209 	Schule 	school
210 	wachsen 	grow
211 	Studie 	study
212 	noch 	still
213 	lernen 	learn
214 	Anlage 	plant
215 	Abdeckung 	cover
216 	Lebensmittel 	food
217 	Sonne 	sun
218 	vier 	four
219 	zwischen 	between
220 	Zustand 	state
221 	halten 	keep
222 	Auge 	eye
223 	nie 	never
224 	letzte 	last
225 	lassen 	let
226 	Gedanken 	thought
227 	Stadt 	city
228 	Baum 	tree
229 	überqueren 	cross
230 	Bauernhof 	farm
231 	schwer 	hard
232 	Beginn 	start
233 	Macht 	might
234 	Geschichte 	story
235 	Säge 	saw
236 	weit 	far
237 	Meer 	sea
238 	ziehen 	draw
239 	links 	left
240 	spät 	late
241 	laufen 	run
242 	unterlassen Sie 	don’t
243 	während 	while
244 	Presse 	press
245 	Schließen 	close
246 	Nacht 	night
247 	realen 	real
248 	Leben 	life
249 	wenige 	few
250 	Norden 	north
251 	Buch 	book
252 	tragen 	carry
253 	nahm 	took
254 	Wissenschaft 	science
255 	essen 	eat
256 	Zimmer 	room
257 	Freund 	friend
258 	begann 	began
259 	Idee 	idea
260 	Fisch 	fish
261 	berg 	mountain
262 	Stopp 	stop
263 	einmal 	once
264 	Basis 	base
265 	hören 	hear
266 	Pferd 	horse
267 	Schnitt 	cut
268 	sicher 	sure
269 	beobachten 	watch
270 	Farbe 	color
271 	Gesicht 	face
272 	Holz 	wood
273 	Haupt- 	main
274 	geöffnet 	open
275 	scheinen 	seem
276 	zusammen 	together
277 	nächste 	next
278 	weiß 	white
279 	Kinder 	children
280 	Start 	begin
281 	bekam 	got
282 	gehen 	walk
283 	Beispiel 	example
284 	erleichtern 	ease
285 	Papier 	paper
286 	Gruppe 	group
287 	immer 	always
288 	Musik 	music
289 	diejenigen 	those
290 	beide 	both
291 	Marke 	mark
292 	oft 	often
293 	Schreiben 	letter
294 	bis 	until
295 	Meile 	mile
296 	Fluss 	river
297 	Auto 	car
298 	Füße 	feet
299 	Pflege 	care
300 	zweite 	second
301 	genug 	enough
302 	Ebene 	plain
303 	Mädchen 	girl
304 	üblich 	usual
305 	jung 	young
306 	bereit 	ready
307 	oben 	above
308 	je 	ever
309 	rot 	red
310 	Liste 	list
311 	obwohl 	though
312 	fühlen 	feel
313 	Vortrag 	talk
314 	Vogel 	bird
315 	bald 	soon
316 	Körper 	body
317 	Hund 	dog
318 	Familie 	family
319 	direkt 	direct
320 	Pose 	pose
321 	verlassen 	leave
322 	Lied 	song
323 	messen 	measure
324 	Tür 	door
325 	Produkt 	product
326 	schwarz 	black
327 	kurz 	short
328 	Zahl 	numeral
329 	Klasse 	class
330 	Wind 	wind
331 	Frage 	question
332 	passieren 	happen
333 	vollständig 	complete
334 	Schiff 	ship
335 	Bereich 	area
336 	Hälfte 	half
337 	Stein 	rock
338 	bestellen 	order
339 	Feuer 	fire
340 	Süden 	south
341 	Problem 	problem
342 	Stück 	piece
343 	sagte 	told
344 	wusste 	knew
345 	passieren 	pass
346 	seit 	since
347 	obere 	top
348 	ganze 	whole
349 	König 	king
350 	Straße 	street
351 	Zoll 	inch
352 	multiplizieren 	multiply
353 	nichts 	nothing
354 	Kurs 	course
355 	bleiben 	stay
356 	Rad 	wheel
357 	voll 	full
358 	Kraft 	force
359 	blau 	blue
360 	Objekt 	object
361 	entscheiden 	decide
362 	Oberfläche 	surface
363 	tief 	deep
364 	Mond 	moon
365 	Insel 	island
366 	Fuß 	foot
367 	System 	system
368 	beschäftigt 	busy
369 	Prüfung 	test
370 	Rekord 	record
371 	Boot 	boat
372 	gemeinsam 	common
373 	goldenen 	gold
374 	möglich 	possible
375 	Flugzeug 	plane
376 	statt 	stead
377 	trocken 	dry
378 	Wunder 	wonder
379 	Lachen 	laugh
380 	tausend 	thousand
381 	vor 	ago
382 	lief 	ran
383 	überprüfen 	check
384 	Spiel 	game
385 	Form 	shape
386 	gleichsetzen 	equate
387 	heiß 	hot
388 	Fehl 	miss
389 	gebracht 	brought
390 	Wärme 	heat
391 	Schnee 	snow
392 	Reifen 	tire
393 	bringen 	bring
394 	ja 	yes
395 	entfernt 	distant
396 	füllen 	fill
397 	Osten 	east
398 	malen 	paint
399 	Sprache 	language
400 	unter 	among
401 	Einheit 	unit
402 	Macht 	power
403 	Stadt 	town
404 	fein 	fine
405 	sicher 	certain
406 	fliegen 	fly
407 	fallen 	fall
408 	führen 	lead
409 	Schrei 	cry
410 	dunkel 	dark
411 	Maschine 	machine
412 	note 	note
413 	warten 	wait
414 	Plan 	plan
415 	Abbildung 	figure
416 	Stern 	star
417 	Kasten 	box
418 	Nomen 	noun
419 	Feld 	field
420 	Rest 	rest
421 	richtig 	correct
422 	fähig 	able
423 	Pfund 	pound
424 	getan 	done
425 	Schönheit 	beauty
426 	Antriebs 	drive
427 	stand 	stood
428 	enthalten 	contain
429 	Front 	front
430 	lehren 	teach
431 	Woche 	week
432 	Finale 	final
433 	gab 	gave
434 	grün 	green
435 	oh 	oh
436 	schnell 	quick
437 	entwickeln 	develop
438 	Ozean 	ocean
439 	warme 	warm
440 	kostenlos 	free
441 	Minute 	minute
442 	stark 	strong
443 	besondere 	special
444 	Geist 	mind
445 	hinter 	behind
446 	klar 	clear
447 	Schwanz 	tail
448 	produzieren 	produce
449 	Tatsache 	fact
450 	Raum 	space
451 	gehört 	heard
452 	beste 	best
453 	Stunde 	hour
454 	besser 	better
455 	wahr 	true
456 	während 	during
457 	hundert 	hundred
458 	fünf 	five
459 	merken 	remember
460 	Schritt 	step
461 	früh 	early
462 	halten 	hold
463 	Westen 	west
464 	Boden 	ground
465 	Interesse 	interest
466 	erreichen 	reach
467 	schnell 	fast
468 	Verbum 	verb
469 	singen 	sing
470 	hören 	listen
471 	sechs 	six
472 	Tabelle 	table
473 	Reise 	travel
474 	weniger 	less
475 	Morgen 	morning
476 	zehn 	ten
477 	einfach 	simple
478 	mehrere 	several
479 	Vokal 	vowel
480 	auf 	toward
481 	Krieg 	war
482 	legen 	lay
483 	gegen 	against
484 	Muster 	pattern
485 	schleppend 	slow
486 	Zentrum 	center
487 	Liebe 	love
488 	Person 	person
489 	Geld 	money
490 	dienen 	serve
491 	erscheinen 	appear
492 	Straße 	road
493 	Karte 	map
494 	regen 	rain
495 	Regel 	rule
496 	regieren 	govern
497 	ziehen 	pull
498 	Kälte 	cold
499 	Hinweis 	notice
500 	Stimme 	voice
501 	Energie 	energy
502 	Jagd 	hunt
503 	wahrscheinlich 	probable
504 	Bett 	bed
505 	Bruder 	brother
506 	Ei 	egg
507 	Fahrt 	ride
508 	Zelle 	cell
509 	glauben 	believe
510 	vielleicht 	perhaps
511 	pflücken 	pick
512 	plötzlich 	sudden
513 	zählen 	count
514 	Platz 	square
515 	Grund 	reason
516 	Dauer 	length
517 	vertreten 	represent
518 	Kunst 	art
519 	Thema 	subject
520 	Region 	region
521 	Größe 	size
522 	variieren 	vary
523 	regeln 	settle
524 	sprechen 	speak
525 	Gewicht 	weight
526 	allgemein 	general
527 	Eis 	ice
528 	Materie 	matter
529 	Kreis 	circle
530 	Paar 	pair
531 	umfassen 	include
532 	Kluft 	divide
533 	Silbe 	syllable
534 	Filz 	felt
535 	groß 	grand
536 	Kugel 	ball
537 	noch 	yet
538 	Welle 	wave
539 	fallen 	drop
540 	Herz 	heart
541 	Uhr 	am
542 	vorhanden 	present
543 	schwer 	heavy
544 	Tanz 	dance
545 	Motor 	engine
546 	Position 	position
547 	Arm 	arm
548 	breit 	wide
549 	Segel 	sail
550 	Material 	material
551 	Fraktion 	fraction
552 	Wald 	forest
553 	sitzen 	sit
554 	Rennen 	race
555 	Fenster 	window
556 	Speicher 	store
557 	Sommer 	summer
558 	Zug 	train
559 	Schlaf 	sleep
560 	beweisen 	prove
561 	einsam 	lone
562 	Bein 	leg
563 	Übung 	exercise
564 	Wand 	wall
565 	Fang 	catch
566 	Berg 	mount
567 	wünschen 	wish
568 	Himmel 	sky
569 	Board 	board
570 	Freude 	joy
571 	Winter 	winter
572 	sa 	sat
573 	geschrieben 	written
574 	wilden 	wild
575 	Instrument 	instrument
576 	gehalten 	kept
577 	Glas 	glass
578 	Gras 	grass
579 	Kuh 	cow
580 	Arbeit 	job
581 	Rand 	edge
582 	Zeichen 	sign
583 	Besuch 	visit
584 	Vergangenheit 	past
585 	weich 	soft
586 	Spaß 	fun
587 	hell 	bright
588 	Gases 	gas
589 	Wetter 	weather
590 	Monat 	month
591 	Million 	million
592 	tragen 	bear
593 	Finish 	finish
594 	glücklich 	happy
595 	hoffen 	hope
596 	blume 	flower
597 	kleiden 	clothe
598 	seltsam 	strange
599 	Vorbei 	gone
600 	Handel 	trade
601 	Melodie 	melody
602 	Reise 	trip
603 	Büro 	office
604 	empfangen 	receive
605 	Reihe 	row
606 	Mund 	mouth
607 	genau 	exact
608 	Zeichen 	symbol
609 	sterben 	die
610 	am wenigsten 	least
611 	Ärger 	trouble
612 	Schrei 	shout
613 	außer 	except
614 	schrieb 	wrote
615 	Samen 	seed
616 	Ton 	tone
617 	beitreten 	join
618 	vorschlagen 	suggest
619 	sauber 	clean
620 	Pause 	break
621 	Dame 	lady
622 	Hof 	yard
623 	steigen 	rise
624 	schlecht 	bad
625 	Schlag 	blow
626 	Öl 	oil
627 	Blut 	blood
628 	berühren 	touch
629 	wuchs 	grew
630 	Cent 	cent
631 	mischen 	mix
632 	Mannschaft 	team
633 	Draht 	wire
634 	Kosten 	cost
635 	verloren 	lost
636 	braun 	brown
637 	tragen 	wear
638 	Garten 	garden
639 	gleich 	equal
640 	gesendet 	sent
641 	wählen 	choose
642 	fiel 	fell
643 	passen 	fit
644 	fließen 	flow
645 	Messe 	fair
646 	Bank 	bank
647 	sammeln 	collect
648 	sparen 	save
649 	Kontrolle 	control
650 	dezimal 	decimal
651 	Ohr 	ear
652 	sonst 	else
653 	ganz 	quite
654 	pleite 	broke
655 	Fall 	case
656 	Mitte 	middle
657 	töten 	kill
658 	Sohn 	son
659 	See 	lake
660 	Moment 	moment
661 	Maßstab 	scale
662 	laut 	loud
663 	Frühling 	spring
664 	beobachten 	observe
665 	Kind 	child
666 	gerade 	straight
667 	Konsonant 	consonant
668 	Nation 	nation
669 	Wörterbuch 	dictionary
670 	milch 	milk
671 	Geschwindigkeit 	speed
672 	Verfahren 	method
673 	Orgel 	organ
674 	zahlen 	pay
675 	Alter 	age
676 	Abschnitt 	section
677 	Kleid 	dress
678 	Wolke 	cloud
679 	Überraschung 	surprise
680 	ruhig 	quiet
681 	Stein 	stone
682 	winzig 	tiny
683 	Aufstieg 	climb
684 	kühlen 	cool
685 	Entwurf 	design
686 	arm 	poor
687 	Menge 	lot
688 	Versuch 	experiment
689 	Boden 	bottom
690 	Schlüssel 	key
691 	Eisen 	iron
692 	Einzel 	single
693 	Stick 	stick
694 	Wohnung 	flat
695 	zwanzig 	twenty
696 	Haut 	skin
697 	Lächeln 	smile
698 	Falte 	crease
699 	Loch 	hole
700 	springen 	jump
701 	Kind 	baby
702 	acht 	eight
703 	Dorf 	village
704 	treffen 	meet
705 	Wurzel 	root
706 	kaufen 	buy
707 	erhöhen 	raise
708 	lösen 	solve
709 	Metall 	metal
710 	ob 	whether
711 	drücken 	push
712 	sieben 	seven
713 	Absatz 	paragraph
714 	dritte 	third
715 	wird 	shall
716 	Hand 	held
717 	Haar 	hair
718 	beschreiben 	describe
719 	Koch 	cook
720 	Boden 	floor
721 	entweder 	either
722 	Ergebnis 	result
723 	brennen 	burn
724 	Hügel 	hill
725 	sicher 	safe
726 	Katze 	cat
727 	Jahrhundert 	century
728 	betrachten 	consider
729 	Typ 	type
730 	Gesetz 	law
731 	Bit 	bit
732 	Küste 	coast
733 	Kopie 	copy
734 	Ausdruck 	phrase
735 	still 	silent
736 	hoch 	tall
737 	Sand 	sand
738 	Boden 	soil
739 	Rolle 	roll
740 	Temperatur 	temperature
741 	Finger 	finger
742 	Industrie 	industry
743 	Wert 	value
744 	Kampf 	fight
745 	Lüge 	lie
746 	schlagen 	beat
747 	begeistern 	excite
748 	natürlich 	natural
749 	Blick 	view
750 	Sinn 	sense
751 	Hauptstadt 	capital
752 	wird nicht 	won’t
753 	Stuhl 	chair
754 	Achtung 	danger
755 	Obst 	fruit
756 	reich 	rich
757 	dick 	thick
758 	Soldat 	soldier
759 	Prozess 	process
760 	betreiben 	operate
761 	Praxis 	practice
762 	trennen 	separate
763 	schwierig 	difficult
764 	Arzt 	doctor
765 	Bitte 	please
766 	schützen 	protect
767 	Mittag 	noon
768 	Ernte 	crop
769 	modernen 	modern
770 	Elementes 	element
771 	treffen 	hit
772 	Schüler 	student
773 	Ecke 	corner
774 	Partei 	party
775 	Versorgung 	supply
776 	deren 	whose
777 	lokalisieren 	locate
778 	Rings 	ring
779 	Charakter 	character
780 	insekt 	insect
781 	gefangen 	caught
782 	Zeit 	period
783 	zeigen 	indicate
784 	Funk 	radio
785 	Speiche 	spoke
786 	Atom 	atom
787 	Mensch 	human
788 	Geschichte 	history
789 	Wirkung 	effect
790 	elektrisch 	electric
791 	erwarten 	expect
792 	Knochen 	bone
793 	Schiene 	rail
794 	vorstellen 	imagine
795 	bieten 	provide
796 	zustimmen 	agree
797 	so 	thus
798 	sanft 	gentle
799 	Frau 	woman
800 	Kapitän 	captain
801 	erraten 	guess
802 	erforderlich 	necessary
803 	scharf 	sharp
804 	Flügel 	wing
805 	schaffen 	create
806 	Nachbar 	neighbor
807 	Wasch 	wash
808 	Fledermaus 	bat
809 	eher 	rather
810 	Menge 	crowd
811 	mais 	corn
812 	vergleichen 	compare
813 	Gedicht 	poem
814 	Schnur 	string
815 	Glocke 	bell
816 	abhängen 	depend
817 	Fleisch 	meat
818 	einreiben 	rub
819 	Rohr 	tube
820 	berühmt 	famous
921 	Dollar 	dollar
822 	Strom 	stream
823 	Angst 	fear
284 	Blick 	sight
825 	dünn 	thin
826 	Dreieck 	triangle
827 	Erde 	planet
828 	Eile 	hurry
829 	Chef 	chief
830 	Kolonie 	colony
831 	Uhr 	clock
832 	Mine 	mine
833 	Krawatte 	tie
834 	eingeben 	enter
835 	Dur 	major
836 	frisch 	fresh
837 	Suche 	search
838 	senden 	send
839 	gelb 	yellow
840 	Pistole 	gun
841 	erlauben 	allow
842 	Druck 	print
843 	tot 	dead
844 	Stelle 	spot
845 	Wüste 	desert
846 	Anzug 	suit
847 	Strom 	current
848 	Aufzug 	lift
840 	stiegen 	rose
850 	ankommen 	arrive
851 	Stamm 	master
852 	Spur 	track
853 	Elternteil 	parent
854 	Ufer 	shore
855 	Teilung 	division
856 	Blatt 	sheet
857 	Substanz 	substance
858 	begünstigen 	favor
859 	verbinden 	connect
860 	nach 	post
861 	verbringen 	spend
862 	Akkord 	chord
863 	Fett 	fat
864 	froh 	glad
865 	Original 	original
866 	Aktie 	share
867 	Station 	station
868 	Papa 	dad
869 	Brot 	bread
870 	aufladen 	charge
871 	richtig 	proper
872 	Leiste 	bar
873 	Angebot 	offer
874 	Segment 	segment
875 	Sklave 	slave
876 	ente 	duck
877 	Augenblick 	instant
878 	Markt 	market
879 	Grad 	degree
880 	besiedeln 	populate
881 	küken 	chick
882 	liebe 	dear
883 	Feind 	enemy
884 	antworten 	reply
885 	Getränk 	drink
886 	auftreten 	occur
887 	Unterstützung 	support
888 	Rede 	speech
889 	Natur 	nature
890 	Angebot 	range
891 	Dampf 	steam
892 	Bewegung 	motion
893 	Weg 	path
894 	Flüssigkeit 	liquid
895 	protokollieren 	log
896 	gemeint 	meant
897 	Quotient 	quotient
898 	Gebiss 	teeth
899 	Schale 	shell
900 	Hals 	neck
901 	Sauerstoff 	oxygen
902 	Zucker 	sugar
903 	Tod 	death
904 	ziemlich 	pretty
905 	Geschicklichkeit 	skill
906 	Frauen 	women
907 	Saison 	season
908 	Lösung 	solution
909 	Magnet 	magnet
910 	Silber 	silver
911 	danken 	thank
912 	Zweig 	branch
913 	Spiel 	match
914 	Suffix 	suffix
915 	insbesondere 	especially
916 	Feige 	fig
917 	ängstlich 	afraid
918 	riesig 	huge
919 	Schwester 	sister
920 	Stahl 	steel
921 	diskutieren 	discuss
922 	vorwärts 	forward
923 	ähnlich 	similar
924 	führen 	guide
925 	Erfahrung 	experience
926 	Partitur 	score
927 	apfel 	apple
928 	gekauft 	bought
929 	geführt 	led
930 	Tonhöhe 	pitch
931 	Mantel 	coat
932 	Masse 	mass
933 	Karte 	card
934 	Band 	band
935 	Seil 	rope
936 	Rutsch 	slip
937 	gewinnen 	win
938 	träumen 	dream
939 	Abend 	evening
940 	Zustand 	condition
941 	Futtermittel 	feed
942 	Werkzeug 	tool
943 	gesamt 	total
944 	Basis 	basic
945 	Geruch 	smell
946 	Tal 	valley
947 	noch 	nor
948 	doppelt 	double
949 	Sitz 	seat
950 	fortsetzen 	continue
951 	Block 	block
952 	Tabelle 	chart
953 	Hut 	hat
954 	verkaufen 	sell
955 	Erfolg 	success
956 	Firma 	company
957 	subtrahieren 	subtract
958 	Veranstaltung 	event
959 	besondere 	particular
960 	viel 	deal
961 	schwimmen 	swim
962 	Begriff 	term
963 	Gegenteil 	opposite
964 	Frau 	wife
965 	Schuh 	shoe
966 	Schulter 	shoulder
967 	Verbreitung 	spread
968 	arrangieren 	arrange
969 	Lager 	camp
970 	erfinden 	invent
971 	Baumwolle 	cotton
972 	geboren 	born
973 	bestimmen 	determine
974 	Quart 	quart
975 	neun 	nine
976 	Lastwagen 	truck
977 	Lärm 	noise
978 	Ebene 	level
979 	Chance 	chance
980 	sammeln 	gather
981 	Geschäft 	shop
982 	Stretch 	stretch
983 	werfen 	throw
984 	Glanz 	shine
985 	Immobilien 	property
986 	Spalte 	column
987 	Molekül 	molecule
988 	wählen 	select
989 	falsch 	wrong
990 	grau 	gray
991 	Wiederholung 	repeat
992 	erfordern 	require
993 	breit 	broad
994 	vorbereiten 	prepare
995 	Salz 	salt
996 	Nase 	nose
997 	mehreren 	plural
998 	Zorn 	anger
999 	Anspruch 	claim
1000 	Kontinent 	continent`

// Pairs is a list of word pairshttps://strommeninc.com/1000-most-common-german-words-frequency-vocabulary/
var Pairs = []Pair{}

func init() {
	parts := strings.Split(data, "\n")
	if len(parts) != 1000 {
		panic("not enough words")
	}
	for _, part := range parts {
		words := strings.Split(part, " ")
		if len(words) != 3 {
			continue
		}
		Pairs = append(Pairs, Pair{
			German:  words[1],
			English: words[2],
		})
	}
	fmt.Println("number of word pairs is", len(Pairs))
}
